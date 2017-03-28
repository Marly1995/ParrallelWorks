//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
__kernel void reduce_max(__global const int *A, __global int *B, __local int *scratch)
{ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			if(scratch[lid] < scratch[lid+i])
				scratch[lid] = scratch[lid+i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		atomic_max(&B[0],scratch[lid]);
	}
}
__kernel void add(__global const int* A, __global const int* B, __global int* C) 
{
	int id = get_global_id(0);
	int loc_id = get_local_id(0);
	printf("global id = %d, local id = %d\n", id, loc_id); //do it for each work item
	if (id == 0)
	{ 
		printf("work group size %d\n", get_local_size(0));
	}
	C[id] = A[id] + B[id];
}

__kernel void multiply(__global const int* A, __global const int* B, __global int* C)
{ 
	int id = get_global_id(0);
	C[id] = A[id] * B[id];
}

//a simple smoothing kernel averaging values in a local window (radius 1)
__kernel void avg_filter(__global const int* A, __global int* B) 
{
	int id = get_global_id(0);
	B[id] = (A[id - 1] + A[id] + A[id + 1])/3;
}

//a simple 2D kernel
__kernel void add2D(__global const int* A, __global const int* B, __global int* C) 
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int id = x + y*width;

	printf("id = %d x = %d y = %d w = %d h = %d\n", id, x, y, width, height);

	C[id]= A[id]+ B[id];
}

