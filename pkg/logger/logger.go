package logger

import (
	"fmt"
	"log"
	"os"
	"time"
)

// Level represents a log severity level
type Level int

const (
	INFO Level = iota
	WARN
	ERROR
	DEBUG
)

var (
	infoColor  = "\033[1;34m" // Blue
	warnColor  = "\033[1;33m" // Yellow
	errorColor = "\033[1;31m" // Red
	debugColor = "\033[0;36m" // Cyan
	resetColor = "\033[0m"
)

// Logger is a lightweight structured logger
type Logger struct {
	out    *log.Logger
	prefix string
	level  Level
}

// New creates a new Logger instance
func New(prefix string, level Level) *Logger {
	return &Logger{
		out:    log.New(os.Stdout, "", 0),
		prefix: prefix,
		level:  level,
	}
}

// format formats the log message with timestamp and color
func (l *Logger) format(level Level, msg string) string {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	var color, levelStr string

	switch level {
	case INFO:
		color, levelStr = infoColor, "INFO"
	case WARN:
		color, levelStr = warnColor, "WARN"
	case ERROR:
		color, levelStr = errorColor, "ERROR"
	case DEBUG:
		color, levelStr = debugColor, "DEBUG"
	}

	return fmt.Sprintf("%s[%s] [%s] [%s]%s %s \n",
		color, timestamp, levelStr, l.prefix, resetColor, msg)
}

// Info logs an informational message
func (l *Logger) Info(format string, a ...interface{}) {
	if l.level <= INFO {
		l.out.Println(l.format(INFO, fmt.Sprintf(format, a...)))
	}
}

// Warn logs a warning message
func (l *Logger) Warn(format string, a ...interface{}) {
	if l.level <= WARN {
		l.out.Println(l.format(WARN, fmt.Sprintf(format, a...)))
	}
}

// Error logs an error message
func (l *Logger) Error(format string, a ...interface{}) {
	if l.level <= ERROR {
		l.out.Println(l.format(ERROR, fmt.Sprintf(format, a...)))
	}
}

// Debug logs a debug message
func (l *Logger) Debug(format string, a ...interface{}) {
	if l.level <= DEBUG {
		l.out.Println(l.format(DEBUG, fmt.Sprintf(format, a...)))
	}
}
