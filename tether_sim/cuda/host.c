/* Standard C11 host API. Caller owns device buffers and CUDA context. */
#include <cuda.h>
#include <stddef.h>
static CUmodule module;
static CUfunction function;
int tether_init(const char *path) {
    CUresult result=cuModuleLoad(&module,path);
    if(result!=CUDA_SUCCESS)return (int)result;
    return (int)cuModuleGetFunction(&function,module,"step_kernel");
}
int tether_step(double *p,double *v,float *q,float *motors,const float *a,
 const float *mass,const float *params,const float *wind,const float *boxes,int *failed,
 int n,int k,float dt,int steps,int iterations,void *stream) {
    if(!function||n<1||k<0||dt<=0||steps<1||iterations<1)return CUDA_ERROR_INVALID_VALUE;
    void *args[]={&p,&v,&q,&motors,&a,&mass,&params,&wind,&boxes,&failed,&n,&k,&dt,&steps,&iterations};
    return (int)cuLaunchKernel(function,(n+127)/128,1,1,128,1,1,0,(CUstream)stream,args,NULL);
}
int tether_close(void) {
    if(!module)return 0;
    CUresult result=cuModuleUnload(module); module=NULL; function=NULL;
    return (int)result;
}
