/* SPDX-License-Identifier: Apache-2.0
 *
 * DEVIATION 2518: in-process native backtrace of a hung thread, no ptrace.
 *
 * On the RTX 4090 RunPod image the byte LM lifetime harness
 * (tools/byte_lm_lifetime_diag.py) hangs inside its second native call, gdb
 * cannot attach ("ptrace: Inappropriate ioctl for device", run 5,
 * bench/results/e1g/2026-09-11_002601-nvidia) and py-spy is not installable.
 * So the process has to produce its own native stack.
 *
 * This library is LD_PRELOADed into each harness child. Its constructor does
 * exactly one thing when MOJOLEARN_NATIVE_STACK_FILE is set: it installs a
 * SIGUSR2 handler. Nothing else changes in the process (no hooks, no wrapped
 * symbols, no threads). When the constructor's environment variable is unset
 * it returns without installing anything.
 *
 * The handler appends, to the file named by MOJOLEARN_NATIVE_STACK_FILE:
 *
 *   === native stack sample pid <pid> tid <tid> time <sec>.<nsec>
 *   [interrupted pc/sp on x86_64 and aarch64, from the ucontext]
 *   <glibc backtrace() of the receiving thread, backtrace_symbols_fd form:
 *    module(symbol+offset) [address]; stripped modules give module(+offset)>
 *   --- /proc/self/task/<tid>/wchan: ...
 *   --- /proc/self/task/<tid>/syscall: ...
 *   --- /proc/self/task/<tid>/stack: ... (usually EACCES without CAP_SYS_ADMIN)
 *   === end sample
 *
 * and RETURNS, so an interrupted futex/condition wait resumes (SA_RESTART).
 * Three samples two seconds apart show whether the frames move. The
 * backtrace passes through the signal trampoline (glibc marks it as a signal
 * frame), so the frames below the handler are the interrupted call chain.
 *
 * Caveat, recorded here so the reader does not over-read the /proc lines:
 * the wchan/syscall lines are read BY the receiving thread ABOUT ITSELF while
 * it is running the handler, so they describe the handler's own state
 * ("running", or the read(2) in flight), not the interrupted wait. The
 * interrupted wait is what the backtrace and the pc show; the parent's
 * /proc/<pid>/task/<tid>/{wchan,syscall} snapshot (taken from outside while
 * the thread is blocked) is the authoritative per-thread kernel state.
 *
 * Async-signal safety: only write(2), open(2), read(2), close(2),
 * clock_gettime, getpid, syscall(SYS_gettid), backtrace and
 * backtrace_symbols_fd are used inside the handler. glibc's backtrace loads
 * libgcc_s lazily on its FIRST call (a dlopen, not signal-safe), so the
 * constructor calls it once at load. One residual hazard: the unwinder
 * takes the loader's read lock in dl_iterate_phdr; a thread hung INSIDE the
 * dynamic loader would deadlock in the handler. The parent still kills the
 * child at its deadline, so the worst case is a missing sample, and the
 * header line (written before the backtrace) says the signal was delivered.
 *
 * Build (on the box; tools/byte_lm_lifetime_diag.sh does this):
 *   cc -shared -fPIC -O1 -g -o native_stack_dump.so tools/native_stack_dump.c
 * glibc only (execinfo.h). Linux only (/proc, SYS_gettid).
 */
#define _GNU_SOURCE
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/ucontext.h>
#endif

#define NSD_MAX_FRAMES 256
#define NSD_PATH_MAX 4096

static char nsd_path[NSD_PATH_MAX];
static int nsd_installed = 0;

static unsigned long long nsd_gettid(void) {
#if defined(__linux__)
    return (unsigned long long)syscall(SYS_gettid);
#else
    return 0ULL; /* Linux only; the Mac compile check never runs this */
#endif
}

/* write(2) everything, tolerating short writes and EINTR. */
static void nsd_write_all(int fd, const char *buf, size_t len) {
    while (len > 0) {
        ssize_t n = write(fd, buf, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return;
        }
        buf += (size_t)n;
        len -= (size_t)n;
    }
}

static void nsd_puts(int fd, const char *s) {
    nsd_write_all(fd, s, strlen(s));
}

/* Decimal, no allocation. */
static void nsd_put_dec(int fd, unsigned long long v) {
    char buf[32];
    int i = (int)sizeof buf;
    buf[--i] = '\0';
    if (v == 0) buf[--i] = '0';
    while (v > 0 && i > 0) {
        buf[--i] = (char)('0' + (v % 10));
        v /= 10;
    }
    nsd_puts(fd, buf + i);
}

/* Zero-padded decimal of fixed width (for nanoseconds). */
static void nsd_put_dec_pad(int fd, unsigned long long v, int width) {
    char buf[32];
    int i = (int)sizeof buf;
    buf[--i] = '\0';
    while (width-- > 0 && i > 0) {
        buf[--i] = (char)('0' + (v % 10));
        v /= 10;
    }
    nsd_puts(fd, buf + i);
}

__attribute__((unused)) static void nsd_put_hex(int fd, unsigned long long v) {
    static const char digits[] = "0123456789abcdef";
    char buf[32];
    int i = (int)sizeof buf;
    buf[--i] = '\0';
    if (v == 0) buf[--i] = '0';
    while (v > 0 && i > 0) {
        buf[--i] = digits[v & 0xf];
        v >>= 4;
    }
    nsd_puts(fd, "0x");
    nsd_puts(fd, buf + i);
}

/* Copy a small /proc file to fd, or the errno of the failed open/read. */
static void nsd_copy_proc(int fd, const char *prefix, const char *tail, unsigned long long tid) {
    char path[128];
    char buf[4096];
    size_t pos = 0;
    const char *head = "/proc/self/task/";
    size_t hl = strlen(head), tl = strlen(tail);
    char tidbuf[32];
    int ti = (int)sizeof tidbuf;
    unsigned long long v = tid;
    tidbuf[--ti] = '\0';
    if (v == 0) tidbuf[--ti] = '0';
    while (v > 0 && ti > 0) {
        tidbuf[--ti] = (char)('0' + (v % 10));
        v /= 10;
    }
    if (hl + strlen(tidbuf + ti) + 1 + tl + 1 > sizeof path) return;
    memcpy(path + pos, head, hl); pos += hl;
    memcpy(path + pos, tidbuf + ti, strlen(tidbuf + ti)); pos += strlen(tidbuf + ti);
    path[pos++] = '/';
    memcpy(path + pos, tail, tl); pos += tl;
    path[pos] = '\0';

    nsd_puts(fd, prefix);
    nsd_puts(fd, path);
    nsd_puts(fd, ": ");
    int pfd = open(path, O_RDONLY | O_CLOEXEC);
    if (pfd < 0) {
        nsd_puts(fd, "unreadable errno=");
        nsd_put_dec(fd, (unsigned long long)errno);
        nsd_puts(fd, "\n");
        return;
    }
    size_t total = 0;
    char last = '\0';
    for (;;) {
        ssize_t n = read(pfd, buf, sizeof buf);
        if (n < 0) {
            if (errno == EINTR) continue;
            nsd_puts(fd, "\n(read failed errno=");
            nsd_put_dec(fd, (unsigned long long)errno);
            nsd_puts(fd, ")");
            last = ')';
            break;
        }
        if (n == 0) break;
        nsd_write_all(fd, buf, (size_t)n);
        total += (size_t)n;
        last = buf[n - 1];
        if (total >= 16384) { nsd_puts(fd, "\n(truncated)"); last = ')'; break; }
    }
    close(pfd);
    if (total == 0 || last != '\n') nsd_puts(fd, "\n");
}

static void nsd_handler(int sig, siginfo_t *info, void *uctx) {
    int saved_errno = errno;
    (void)sig;
    int fd = open(nsd_path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
        errno = saved_errno;
        return;
    }
    unsigned long long pid = (unsigned long long)getpid();
    unsigned long long tid = nsd_gettid();
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 0;
    clock_gettime(CLOCK_REALTIME, &ts);

    nsd_puts(fd, "=== native stack sample pid ");
    nsd_put_dec(fd, pid);
    nsd_puts(fd, " tid ");
    nsd_put_dec(fd, tid);
    nsd_puts(fd, " time ");
    nsd_put_dec(fd, (unsigned long long)ts.tv_sec);
    nsd_puts(fd, ".");
    nsd_put_dec_pad(fd, (unsigned long long)ts.tv_nsec, 9);
    nsd_puts(fd, " sender_pid ");
    nsd_put_dec(fd, info ? (unsigned long long)info->si_pid : 0ULL);
    nsd_puts(fd, "\n");

    /* The interrupted program counter and stack pointer, from the ucontext.
     * These name the frame the thread was blocked in even if the unwinder
     * stops early. */
    if (uctx) {
        ucontext_t *uc = (ucontext_t *)uctx;
#if defined(__linux__) && defined(__x86_64__)
        nsd_puts(fd, "interrupted pc ");
        nsd_put_hex(fd, (unsigned long long)uc->uc_mcontext.gregs[REG_RIP]);
        nsd_puts(fd, " sp ");
        nsd_put_hex(fd, (unsigned long long)uc->uc_mcontext.gregs[REG_RSP]);
        nsd_puts(fd, " rax ");
        nsd_put_hex(fd, (unsigned long long)uc->uc_mcontext.gregs[REG_RAX]);
        nsd_puts(fd, "\n");
#elif defined(__linux__) && defined(__aarch64__)
        nsd_puts(fd, "interrupted pc ");
        nsd_put_hex(fd, (unsigned long long)uc->uc_mcontext.pc);
        nsd_puts(fd, " sp ");
        nsd_put_hex(fd, (unsigned long long)uc->uc_mcontext.sp);
        nsd_puts(fd, "\n");
#else
        (void)uc;
        nsd_puts(fd, "interrupted pc: unavailable on this architecture\n");
#endif
    }

    void *frames[NSD_MAX_FRAMES];
    int n = backtrace(frames, NSD_MAX_FRAMES);
    nsd_puts(fd, "backtrace frames ");
    nsd_put_dec(fd, (unsigned long long)(n < 0 ? 0 : n));
    nsd_puts(fd, "\n");
    if (n > 0) backtrace_symbols_fd(frames, n, fd);

    /* Same-thread view; see the file header for what these can and cannot
     * say. Kept because they are free and prove which thread ran. */
    nsd_copy_proc(fd, "--- ", "wchan", tid);
    nsd_copy_proc(fd, "--- ", "syscall", tid);
    nsd_copy_proc(fd, "--- ", "stack", tid);
    nsd_puts(fd, "=== end sample\n");
    close(fd);
    errno = saved_errno;
}

__attribute__((constructor)) static void nsd_install(void) {
    const char *path = getenv("MOJOLEARN_NATIVE_STACK_FILE");
    if (path == NULL || path[0] == '\0') return;
    size_t len = strlen(path);
    if (len >= sizeof nsd_path) return;
    memcpy(nsd_path, path, len + 1);

    /* Prime glibc's lazy libgcc_s load outside any signal context. */
    void *warm[8];
    (void)backtrace(warm, 8);

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = nsd_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGUSR2, &sa, NULL) != 0) return;
    nsd_installed = 1;

    /* One line at load: the harness treats its presence as "the handler is
     * installed in this process" and only then sends SIGUSR2 (the default
     * disposition of SIGUSR2 would terminate the process). */
    int fd = open(nsd_path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return;
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 0;
    clock_gettime(CLOCK_REALTIME, &ts);
    nsd_puts(fd, "=== native stack handler installed pid ");
    nsd_put_dec(fd, (unsigned long long)getpid());
    nsd_puts(fd, " tid ");
    nsd_put_dec(fd, nsd_gettid());
    nsd_puts(fd, " time ");
    nsd_put_dec(fd, (unsigned long long)ts.tv_sec);
    nsd_puts(fd, ".");
    nsd_put_dec_pad(fd, (unsigned long long)ts.tv_nsec, 9);
    nsd_puts(fd, " signal SIGUSR2\n");
    close(fd);
}

/* Read-back for a caller that wants to check without sending a signal. */
int mojolearn_native_stack_dump_installed(void) {
    return nsd_installed;
}
