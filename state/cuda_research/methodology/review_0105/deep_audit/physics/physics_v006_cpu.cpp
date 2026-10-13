// Experimental reduced model, not a MuJoCo-equivalent or calibrated backend.

#include <math.h>
#define B 14
#ifdef __cplusplus
extern "C"
#endif // drone, twelve cable particles, payload
void step_kernel(double* p,double* v,float* q,float* motors,
 const float* actions,const float* mass,const float* params,const float* wind,
 const float* boxes,int* failed,int n,int k,float dt,int steps,int iterations){
 int e=0; if(e>=n||failed[e])return;
 p+=e*B*3; v+=e*B*3; q+=e*7; motors+=e*4; actions+=e*4;
 mass+=e*B; params+=e*9; wind+=e*3; boxes+=e*k*6;
 double old[B*3];
 for(int t=0;t<steps;t++){
  bool contact[B]={false};
  float thrust=0;
  for(int m=0;m<4;m++){motors[m]+=(1-expf(-dt/params[1]))*(fminf(1,fmaxf(0,actions[m]))*params[2]-motors[m]);thrust+=motors[m];}
  float torque[3]={.032f*(motors[0]+motors[1]-motors[2]-motors[3]),
   .032f*(-motors[0]+motors[1]+motors[2]-motors[3]),
   .006f*(-motors[0]+motors[1]-motors[2]+motors[3])};
  float w=q[0],x=q[1],y=q[2],z=q[3];
  float ox=q[4],oy=q[5],oz=q[6];
  float gyro[3]={(params[6]-params[5])*oy*oz,(params[4]-params[6])*oz*ox,(params[5]-params[4])*ox*oy};
  for(int d=0;d<3;d++)q[4+d]+=dt*(torque[d]-gyro[d]-params[7]*q[4+d])/params[4+d];
  ox=q[4];oy=q[5];oz=q[6];
  q[0]+=dt*.5f*(-x*ox-y*oy-z*oz); q[1]+=dt*.5f*(w*ox+y*oz-z*oy);
  q[2]+=dt*.5f*(w*oy+z*ox-x*oz); q[3]+=dt*.5f*(w*oz+x*oy-y*ox);
  float norm=sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
  for(int d=0;d<4;d++)q[d]/=norm;
  // Rotation-first integration: thrust uses the normalized updated attitude.
  w=q[0];x=q[1];y=q[2];z=q[3];
  float axis[3]={2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)};
  for(int b=0;b<B;b++)for(int d=0;d<3;d++){
   int j=b*3+d; old[j]=p[j];
   float force=b==0?axis[d]*thrust+wind[d]-params[3]*v[j]:0;
   v[j]+=dt*(force/mass[b]-(d==2?9.81f:0)); p[j]+=dt*v[j];
  }
  // Position projection: fixed-length particle links; approximate flexible rope.
  for(int it=0;it<iterations;it++){
   for(int b=0;b<B-1;b++){
   double a[3],length=0; for(int d=0;d<3;d++){a[d]=p[(b+1)*3+d]-p[b*3+d];length+=a[d]*a[d];}
   length=sqrt(length); if(length<1e-12f)continue;
   // Tension-only surrogate: slack links exert no compressive projection.
   if(length<=params[0]/(B-1))continue;
   double c=(length-params[0]/(B-1))/length;
   float wa=1/mass[b],wb=1/mass[b+1];
   for(int d=0;d<3;d++){p[b*3+d]+=c*a[d]*wa/(wa+wb);p[(b+1)*3+d]-=c*a[d]*wb/(wa+wb);}
  }
   // Joint projection: enforce floor constraints between rope sweeps.
   if(params[8]>0)for(int b=0;b<B;b++){
    double radius=b==0?.05f:(b==B-1?.025f:.0015f);
    if(p[b*3+2]<radius){p[b*3+2]=radius;contact[b]=true;}
   }
  }
  for(int b=0;b<B;b++){
   float radius=b==0?.05f:(b==B-1?.025f:.0015f);
   for(int d=0;d<3;d++){int j=b*3+d;v[j]=(p[j]-old[j])/dt;if(!isfinite(p[j])||!isfinite(v[j]))failed[e]=2;}
   if(p[b*3+2]<radius||(contact[b]&&p[b*3+2]<=radius)){
    if(params[8]>0){p[b*3+2]=radius;v[b*3+2]=fmax(0.0,v[b*3+2]);double friction=pow(.98,(double)(dt/.001f));v[b*3]*=friction;v[b*3+1]*=friction;}
    else failed[e]=1;
   }
   for(int box=0;box<k;box++){
    float distance=0; for(int d=0;d<3;d++){float h=fmaxf(fabsf(p[b*3+d]-boxes[box*6+d])-boxes[box*6+3+d],0);distance+=h*h;}
    if(distance<radius*radius)failed[e]=1;
   }
  }
  if(failed[e])return;
 }
}


