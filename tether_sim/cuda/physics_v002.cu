// Experimental reduced model, not a MuJoCo-equivalent or calibrated backend.

#include <math.h>
#define B 14
// Exact squared distance from a finite segment to an axis-aligned box.
// Split at slab crossings; each interval has a quadratic distance function.
__device__ static double segment_box_distance2(const double* a,const double* b,const float* box){
 double cut[8]={0,1};int count=2;
 for(int d=0;d<3;d++){
  double delta=b[d]-a[d];if(fabs(delta)<1e-15)continue;
  for(int side=-1;side<=1;side+=2){
   double t=(box[d]+side*box[3+d]-a[d])/delta;
   if(t>0&&t<1)cut[count++]=t;
  }
 }
 for(int i=1;i<count;i++){double t=cut[i];int j=i-1;while(j>=0&&cut[j]>t){cut[j+1]=cut[j];j--;}cut[j+1]=t;}
 double best=1e100;
 for(int i=0;i<count-1;i++){
  double lo=cut[i],hi=cut[i+1],mid=(lo+hi)*.5,aa=0,ab=0;
  for(int d=0;d<3;d++){
   double delta=b[d]-a[d],pos=a[d]+mid*delta;
   double boundary=pos<box[d]-box[3+d]?box[d]-box[3+d]:box[d]+box[3+d];
   if(pos<box[d]-box[3+d]||pos>box[d]+box[3+d]){aa+=delta*delta;ab+=delta*(a[d]-boundary);}
  }
  double candidate=aa>0?fmin(hi,fmax(lo,-ab/aa)):mid;
  double samples[3]={lo,hi,candidate};
  for(int sample=0;sample<3;sample++){
   double distance=0;
   for(int d=0;d<3;d++){double h=fmax(fabs(a[d]+samples[sample]*(b[d]-a[d])-box[d])-box[3+d],0.0);distance+=h*h;}
   best=fmin(best,distance);
  }
 }
 return best;
}
#ifdef __cplusplus
extern "C"
#endif // drone, twelve cable particles, payload
__global__ void step_kernel(double* p,double* v,float* q,float* motors,
 const float* actions,const float* mass,const float* params,const float* wind,
 const float* boxes,int* failed,int n,int k,float dt,int steps,int iterations){
 int e=blockIdx.x*blockDim.x+threadIdx.x; if(e>=n||failed[e])return;
 p+=e*B*3; v+=e*B*3; q+=e*7; motors+=e*4; actions+=e*4;
 mass+=e*B; params+=e*9; wind+=e*3; boxes+=e*k*6;
 double old[B*3];
 for(int t=0;t<steps;t++){
  float thrust=0;
  for(int m=0;m<4;m++){motors[m]+=(1-expf(-dt/params[1]))*(fminf(1,fmaxf(0,actions[m]))*params[2]-motors[m]);thrust+=motors[m];}
  float torque[3]={.032f*(motors[0]+motors[1]-motors[2]-motors[3]),
   .032f*(-motors[0]+motors[1]+motors[2]-motors[3]),
   .006f*(-motors[0]+motors[1]-motors[2]+motors[3])};
  float w=q[0],x=q[1],y=q[2],z=q[3];
  float axis[3]={2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)};
  float ox=q[4],oy=q[5],oz=q[6];
  float gyro[3]={(params[6]-params[5])*oy*oz,(params[4]-params[6])*oz*ox,(params[5]-params[4])*ox*oy};
  for(int d=0;d<3;d++)q[4+d]+=dt*(torque[d]-gyro[d]-params[7]*q[4+d])/params[4+d];
  ox=q[4];oy=q[5];oz=q[6];
  q[0]+=dt*.5f*(-x*ox-y*oy-z*oz); q[1]+=dt*.5f*(w*ox+y*oz-z*oy);
  q[2]+=dt*.5f*(w*oy+z*ox-x*oz); q[3]+=dt*.5f*(w*oz+x*oy-y*ox);
  float norm=sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
  for(int d=0;d<4;d++)q[d]/=norm;
  for(int b=0;b<B;b++)for(int d=0;d<3;d++){
   int j=b*3+d; old[j]=p[j];
   float force=b==0?axis[d]*thrust+wind[d]-params[3]*v[j]:0;
   v[j]+=dt*(force/mass[b]-(d==2?9.81f:0)); p[j]+=dt*v[j];
  }
  // Position projection: fixed-length particle links; approximate flexible rope.
  for(int it=0;it<iterations;it++)for(int b=0;b<B-1;b++){
   double a[3],length=0; for(int d=0;d<3;d++){a[d]=p[(b+1)*3+d]-p[b*3+d];length+=a[d]*a[d];}
   length=sqrt(length); if(length<1e-12f)continue;
   double c=(length-params[0]/(B-1))/length;
   float wa=1/mass[b],wb=1/mass[b+1];
   for(int d=0;d<3;d++){p[b*3+d]+=c*a[d]*wa/(wa+wb);p[(b+1)*3+d]-=c*a[d]*wb/(wa+wb);}
  }
  for(int b=0;b<B;b++){
   float radius=b==0?.05f:(b==B-1?.025f:.0015f);
   for(int d=0;d<3;d++){int j=b*3+d;v[j]=(p[j]-old[j])/dt;if(!isfinite(p[j])||!isfinite(v[j]))failed[e]=2;}
   if(p[b*3+2]<radius){
    if(params[8]>0){p[b*3+2]=radius;v[b*3+2]=fmax(0.0,v[b*3+2]);v[b*3]*=.98;v[b*3+1]*=.98;}
    else failed[e]=1;
   }
   for(int box=0;box<k;box++){
    float distance=0; for(int d=0;d<3;d++){float h=fmaxf(fabsf(p[b*3+d]-boxes[box*6+d])-boxes[box*6+3+d],0);distance+=h*h;}
    if(distance<radius*radius)failed[e]=1;
   }
  }
  // Detection only: finite-radius cable links, including endpoint/grazing contact.
  for(int b=0;b<B-1;b++)for(int box=0;box<k;box++)
   if(segment_box_distance2(p+b*3,p+(b+1)*3,boxes+box*6)<=.0015*.0015)failed[e]=1;
  if(failed[e])return;
 }
}
