CXX = g++
CC = gcc
CXXFLAGS = -Wall -O2 -std=c++11
CFLAGS = -Wall -O2 -std=c99

SRCDIR = src/original_classifier
CLASSDIR = $(SRCDIR)/classifier
OBJDIR = obj
TARGET = bin/mpfs_ga

CPP_SOURCES = $(wildcard $(CLASSDIR)/*.cpp)
CPP_OBJECTS = $(CPP_SOURCES:$(CLASSDIR)/%.cpp=$(OBJDIR)/%.o)

C_SOURCES = $(SRCDIR)/main.c
C_OBJECTS = $(C_SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

all: $(TARGET)

$(TARGET): $(CPP_OBJECTS) $(C_OBJECTS)
	$(CXX) $^ -o $@

$(OBJDIR)/%.o: $(CLASSDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -I$(CLASSDIR) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -I$(CLASSDIR) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean
