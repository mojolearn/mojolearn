#define _GNU_SOURCE
#include <unistd.h>
#include <sched.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>
static int fake(void){ const char*s=getenv("FAKE_NCPU"); return s?atoi(s):0; }
long sysconf(int name){
  static long (*real)(int)=0; if(!real) real=(long(*)(int))dlsym(RTLD_NEXT,"sysconf");
  int f=fake(); if(f && (name==_SC_NPROCESSORS_CONF||name==_SC_NPROCESSORS_ONLN)) return f;
  return real(name);
}
int sched_getaffinity(pid_t pid, size_t sz, cpu_set_t *m){
  static int (*real)(pid_t,size_t,cpu_set_t*)=0; if(!real) real=(int(*)(pid_t,size_t,cpu_set_t*))dlsym(RTLD_NEXT,"sched_getaffinity");
  int f=fake(); if(f){ memset(m,0,sz); for(int i=0;i<f;i++) CPU_SET_S(i,sz,m); return 0; }
  return real(pid,sz,m);
}
