#include <stdio.h>

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

void starttime(void) {}
void stoptime(void) {}
