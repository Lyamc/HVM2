// Portable OS bits used by the HVM C runtime (threads, dynlibs, time, yield).
#ifndef HVM_OS_H
#define HVM_OS_H

#include <stdio.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// windows.h defines VOID / min / max, which collide with HVM's rule tags and helpers.
#ifdef VOID
#undef VOID
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

typedef HANDLE hvm_thread_t;

#define HVM_THREAD_FUNC DWORD WINAPI

static inline int hvm_thread_spawn(hvm_thread_t *out, DWORD (WINAPI *fn)(LPVOID), void *arg) {
  *out = CreateThread(NULL, 0, fn, arg, 0, NULL);
  return *out == NULL ? -1 : 0;
}

static inline int hvm_thread_join(hvm_thread_t t) {
  DWORD rc = WaitForSingleObject(t, INFINITE);
  CloseHandle(t);
  return rc == WAIT_OBJECT_0 ? 0 : -1;
}

#define RTLD_LAZY 0
#define RTLD_NOW  0

static char hvm_dlerr[256];

static inline void *hvm_dlopen(const char *path, int flags) {
  (void)flags;
  SetLastError(0);
  hvm_dlerr[0] = 0;
  HMODULE m = LoadLibraryA(path);
  if (!m) {
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   NULL, GetLastError(), 0, hvm_dlerr, sizeof(hvm_dlerr), NULL);
  }
  return (void *)m;
}

static inline void *hvm_dlsym(void *handle, const char *sym) {
  SetLastError(0);
  hvm_dlerr[0] = 0;
  FARPROC p = GetProcAddress((HMODULE)handle, sym);
  if (!p) {
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   NULL, GetLastError(), 0, hvm_dlerr, sizeof(hvm_dlerr), NULL);
  }
  return (void *)p;
}

static inline int hvm_dlclose(void *handle) {
  return FreeLibrary((HMODULE)handle) ? 0 : -1;
}

static inline char *hvm_dlerror(void) {
  return hvm_dlerr[0] ? hvm_dlerr : NULL;
}

#define dlopen  hvm_dlopen
#define dlsym   hvm_dlsym
#define dlclose hvm_dlclose
#define dlerror hvm_dlerror

static inline void hvm_yield(void) {
  SwitchToThread();
}

static inline unsigned long long hvm_time_ns(void) {
  static LARGE_INTEGER freq;
  static int inited = 0;
  LARGE_INTEGER now;
  if (!inited) {
    QueryPerformanceFrequency(&freq);
    inited = 1;
  }
  QueryPerformanceCounter(&now);
  unsigned long long q = (unsigned long long)now.QuadPart;
  unsigned long long f = (unsigned long long)freq.QuadPart;
  return (q / f) * 1000000000ULL + ((q % f) * 1000000000ULL) / f;
}

// MSVC has no setlinebuf; unbuffered IO is the portable stand-in.
static inline void hvm_setlinebuf(FILE *f) {
  setvbuf(f, NULL, _IONBF, 0);
}

// Use POSIX newlines on pipes so Bend / other tools can split on '\n'.
static inline void hvm_stdio_setup(void) {
#if defined(_MSC_VER) || defined(__MINGW32__)
  _setmode(_fileno(stdout), _O_BINARY);
  _setmode(_fileno(stderr), _O_BINARY);
#endif
  hvm_setlinebuf(stdout);
  hvm_setlinebuf(stderr);
}

static inline void hvm_sleep_ns(unsigned long long ns) {
  if (ns == 0) {
    SwitchToThread();
    return;
  }
  HANDLE timer = CreateWaitableTimerW(NULL, TRUE, NULL);
  if (timer) {
    LARGE_INTEGER due;
    unsigned long long hundred_ns = (ns + 99ULL) / 100ULL;
    if (hundred_ns > 9223372036854775807ULL) {
      hundred_ns = 9223372036854775807ULL;
    }
    due.QuadPart = -(LONGLONG)hundred_ns;
    if (SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE)) {
      WaitForSingleObject(timer, INFINITE);
      CloseHandle(timer);
      return;
    }
    CloseHandle(timer);
  }
  DWORD ms = (DWORD)((ns + 999999ULL) / 1000000ULL);
  if (ms == 0) {
    ms = 1;
  }
  Sleep(ms);
}

#else

#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>

typedef pthread_t hvm_thread_t;

#define HVM_THREAD_FUNC void *

static inline int hvm_thread_spawn(hvm_thread_t *out, void *(*fn)(void *), void *arg) {
  return pthread_create(out, NULL, fn, arg);
}

static inline int hvm_thread_join(hvm_thread_t t) {
  return pthread_join(t, NULL);
}

static inline void hvm_yield(void) {
  sched_yield();
}

static inline unsigned long long hvm_time_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (unsigned long long)ts.tv_sec * 1000000000ULL + (unsigned long long)ts.tv_nsec;
}

static inline void hvm_setlinebuf(FILE *f) {
  setlinebuf(f);
}

static inline void hvm_stdio_setup(void) {
  hvm_setlinebuf(stdout);
  hvm_setlinebuf(stderr);
}

static inline void hvm_sleep_ns(unsigned long long ns) {
  struct timespec ts;
  ts.tv_sec = (time_t)(ns / 1000000000ULL);
  ts.tv_nsec = (long)(ns % 1000000000ULL);
  nanosleep(&ts, NULL);
}

#endif

#endif
