#ifndef QSYS_H
#define QSYS_H

#define QLOG_ERR 3

static inline void qsys_openlog(const char *name) { (void)name; }
static inline void qsyslog(int level, const char *fmt, ...) { (void)level; (void)fmt; }

#endif
