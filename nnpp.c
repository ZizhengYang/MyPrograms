#include <stdio.h>
#include <string.h>

long nodeNum = 0;
long linearNum = 0;

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

void initial_start_zero(int number);
void initial_end_zero(int number);

float doLinearOperation(struct NODE start, struct LINEAR linear);

void main()
{
	
}

void initial_start_zero(int number)
{
	struct NODE start_nodes[number];
	for(int i = 0; i < number; i++)
	{
		struct NODE temp;
		temp.num = nodeNum;
		nodeNum++;
		temp.x = 0;
		start_nodes[i] = temp;
	}
}

void initial_end_zero(int number)
{
	struct NODE end_nodes[number];
	for(int i = 0; i < number; i++)
	{
		struct NODE temp;
		temp.num = nodeNum;
		nodeNum++;
		temp.x = 0;
		end_nodes[i] = temp;
	}
}

float doLinearOperation(struct NODE start, struct LINEAR linear)
{
	float x = start.x;
	float w = linear.w;
	float b = linear.b;
	float y = w * x + b;
	return y;
}
