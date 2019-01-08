#include <stdio.h>
#include <string.h>

long nodeNum = 0;
long linearNum = 0;

void initial_start_zero(int number){};
void initial_end_zero(int number){};

typedef struct NODE{
	long num;	// Serial number of the node
	float x;	// Value of the node
}N;

typedef struct LINEAR{
	long start;	// Starting node's serial number
	long end;	// Ending node's serial number
	long num;	// Serial number of the linear operation
	float w;	// The weight in the linear operation
	float b;	// Thw constant correction of the linear operation
}L;

float doLinearOperation(struct NODE start, struct LINEAR linear);

void main()
{
	
}

void initial_start_zero(int number)
{
	struct N start_nodes[number];
	for(int i = 0; i < number; i++)
	{
		struct N temp_n;
		strcpy( temp_n.num, nodeNum);
		nodeNum++;
		strcpy( temp_n.x, 0);
		start_nodes[i] = temp_n;
	}
}

void initial_end_zero(int number)
{
	struct L end _nodes[number];
	for(int i = 0; i < number; i++)
	{
		
	}
}

float doLinearOperation(struct NODE start, struct LINEAR linear)
{
	x = start.x;
	w = linear.w;
	b = linear.b;
	y = w * x + b;
	return y;
}

