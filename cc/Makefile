CXXFLAGS=-g2
LIBS=-lboost_program_options -lboost_date_time -lboost_filesystem -lboost_system -lpthread
OBJS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))

all: convert

convert: convert.o
	$(CXX) $^ $(LIBS) -o ${@}

clean:
	$(RM) $(OBJS) convert

