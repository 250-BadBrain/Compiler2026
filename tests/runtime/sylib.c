#include <stdio.h>
#include <sys/time.h>
#include <stdarg.h>

static struct timeval sysy_start;
static int sysy_idx = 1;

int getint(void) {
    int x = 0;
    scanf("%d", &x);
    return x;
}

int getch(void) {
    return getchar();
}

float getfloat(void) {
    float x = 0.0f;
    scanf("%f", &x);
    return x;
}

int getarray(int a[]) {
    int n = 0;
    scanf("%d", &n);
    for (int i = 0; i < n; ++i) {
        scanf("%d", &a[i]);
    }
    return n;
}

int getfarray(float a[]) {
    int n = 0;
    scanf("%d", &n);
    for (int i = 0; i < n; ++i) {
        scanf("%f", &a[i]);
    }
    return n;
}

void putint(int x) {
    printf("%d", x);
}

void putch(int x) {
    putchar(x);
}

void putfloat(float x) {
    printf("%a", x);
}

void putarray(int n, int a[]) {
    printf("%d:", n);
    for (int i = 0; i < n; ++i) {
        printf(" %d", a[i]);
    }
    putchar('\n');
}

void putfarray(int n, float a[]) {
    printf("%d:", n);
    for (int i = 0; i < n; ++i) {
        printf(" %a", a[i]);
    }
    putchar('\n');
}

void putf(char format[], ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);
}

void _sysy_starttime(int lineno) {
    (void)lineno;
    gettimeofday(&sysy_start, NULL);
}

void _sysy_stoptime(int lineno) {
    struct timeval end;
    gettimeofday(&end, NULL);
    long elapsed = (end.tv_sec - sysy_start.tv_sec) * 1000000L + (end.tv_usec - sysy_start.tv_usec);
    fprintf(stderr, "Timer#%03d@%04d-%04d: 0H-0M-0S-%ldus\n", sysy_idx, 0, lineno, elapsed);
    fprintf(stderr, "TOTAL: 0H-0M-0S-%ldus\n", elapsed);
    ++sysy_idx;
}
