CC := icc

#CFLAGS := -offload -O3 -openmp
CFLAGS := -no-offload -O3 -openmp
#CFLAGS := -O3 -fopenmp -lm

OBJS := pddp_2means.o

APP := pddp_2means

all: $(APP)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(APP): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(OBJS) $(APP)
