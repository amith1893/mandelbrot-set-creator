#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <time.h>

#define MAX_SOURCE_SIZE (0x100000)


#define CHK(expr) { cl_int _ret=expr; if(_ret){ std::cerr<<"Function call in "<<__FILE__<<", line "<<__LINE__<<" failed with error="<<_ret<<std::endl; abort();}}



void prepare_cmd_queue (cl_device_id &device_id, cl_context &context, cl_command_queue &queue)
{
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;

	//Get platform info
	clGetPlatformIDs (1, &platform_id, &ret_num_platforms);
	
	//Get Device info
	clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
	
	//Create OpenCL context
	context = clCreateContext (NULL, 1, &device_id, NULL, NULL, &ret);

	//Create command queue for that particular device what we want
	queue = clCreateCommandQueue (context, device_id, 0, &ret);
}

void load_program (const char *fname, const cl_device_id device_id, const cl_context &context, cl_program &program, cl_kernel &kernel)
{
	FILE *fp;
	char *source_str;
	size_t source_size;
	cl_int ret;

	//Reading the contents of the opencl file which has 
	//the code/logic to create and execute the 
	//mandelbrot function
	//The contents of the opencl file
	//is called as the kernel source code.
	fp = fopen (fname, "r");
	if (!fp)
	{
		std::cerr << "Failed to load kernel from " << fname << std::endl;
		abort();
	}

	source_str = (char *)malloc (MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose (fp);

	//Create a program from the	kernel source
	program = clCreateProgramWithSource (context, 1, 
				(const char **)&source_str, (const size_t *)&source_size, 
				&ret);

	//Building the program after creating the program 
	//from the kernel source
	clBuildProgram (program, 1, &device_id, NULL, NULL, NULL);

	//Creating the kernel after building the program
	kernel = clCreateKernel (program, "mandelbrot", &ret);
}


int main(int argc, char *argv[]) 
{
	
	double xmin, xmax, ymin, ymax;
	double real_factor, imaginary_factor;
	
	cl_device_id device_id = NULL;
	cl_context context;
	cl_command_queue queue;

	//Command queue created for a single device
	//This is how we can interact with the different
	//devices available in the system.
	prepare_cmd_queue (device_id, context, queue);

	cl_program program;
	cl_kernel mandelbrot;


	load_program ("mandelbrot.cl", device_id, context, program, mandelbrot);	

	/* Analysing the commandline arguments */
	int n = atoi(argv[1]);
	int max_iterations = atoi(argv[2]);
	
	/* Setup xmin, xmax, ymin, ymax */
	xmin = 1.0;
	xmax = 2.0;
	ymin = -2.0;
	ymax = -1.0;

	/* Calculate the real_factor and imaginary_factor */
	real_factor = (xmax - xmin)/n;
	imaginary_factor = (ymax - ymin)/n;
	
	/*Setup data structures*/
	double *real_input = (double *)malloc (sizeof(double) * n * n);
	double *imaginary_input = (double *)malloc (sizeof(double) * n * n);
	int *mandelbrot_output = (int *)malloc (sizeof(int) * n * n);	
	memset (mandelbrot_output, 0, n*n);

	/*Populate the data structures just created - real input*/
	double curr_real_input = xmin;
	real_input[0] = xmin;
	for (int i=1; i<(n*n); i++)
	{
		curr_real_input += real_factor;
		real_input[i] = curr_real_input;
		if ((i+1)%n == 0)
		{
			curr_real_input = xmin;
		}
	}
	

	/*Populate the data structures just created - imaginary input*/
	double curr_imaginary_input = ymin;
	imaginary_input[0] = ymin;
	for (int i=1; i<(n*n); i++)
	{
		imaginary_input[i] = curr_imaginary_input;
		if ((i+1)%n == 0)
		{
			curr_imaginary_input += imaginary_factor;
		}
	}

	clock_t t;
	t = clock();
	/*Create memory buffers on the device for data*/
	cl_int ret;
	size_t byte_sz = n * n * sizeof(int);
	cl_mem r_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, byte_sz, NULL, &ret);
	cl_mem i_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, byte_sz, NULL, &ret);
	cl_mem m_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, byte_sz, NULL, &ret);
	cl_mem max_iter = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);	

	/*Set the arguments of the kernel mandelbrot*/
	clSetKernelArg (mandelbrot, 0, sizeof(r_buffer), (void*) &r_buffer);
	clSetKernelArg (mandelbrot, 1, sizeof(i_buffer), (void*) &i_buffer);
	clSetKernelArg (mandelbrot, 2, sizeof(m_buffer), (void*) &m_buffer);
	clSetKernelArg (mandelbrot, 3, sizeof(max_iter), (void*) &max_iter);

	
	/* Copy the data structures into the buffer */
	CHK (clEnqueueWriteBuffer (queue, r_buffer, CL_TRUE, 0, byte_sz, real_input, 0, NULL, NULL));
	CHK (clEnqueueWriteBuffer (queue, i_buffer, CL_TRUE, 0, byte_sz, imaginary_input, 0, NULL, NULL));
	CHK (clEnqueueWriteBuffer (queue, max_iter, CL_TRUE, 0, sizeof(int), &max_iterations, 0, NULL, NULL));

	
	/*
	 * Execute the OpenCL Kernel now by setting the 
	 * global_item_size and local_item_size
	 */	
	size_t global_item_size = n * n;
	size_t local_item_size = atoi(argv[3]);

	CHK (clEnqueueNDRangeKernel (queue, mandelbrot, 1, NULL, 
							&global_item_size, &local_item_size, 0,
							NULL, NULL));


	CHK (clEnqueueReadBuffer (queue, m_buffer, CL_TRUE, 0, byte_sz, mandelbrot_output, 0, NULL, NULL));

	//printf("X:%lf Y:%lf M:%d\n", real_input[4186116], imaginary_input[4186116], mandelbrot_output[4186116]);
	t = clock() - t;
	double time_taken = ((double)t) / CLOCKS_PER_SEC;	
	printf("Time taken %lf\n", time_taken);	
#if 0
	for (int ii=0; ii<(n*n); ii++)
	{
		printf("X:%lf Y:%lf M:%d\n", real_input[ii], imaginary_input[ii], mandelbrot_output[ii]);
	}
#endif

	clFlush(queue);
	clFinish(queue);
	clReleaseKernel(mandelbrot);
	clReleaseProgram(program);
	clReleaseMemObject(r_buffer);
	clReleaseMemObject(i_buffer);
	clReleaseMemObject(m_buffer);
	clReleaseMemObject(max_iter);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	free(real_input);
	free(imaginary_input);
	free(mandelbrot_output);

	return 0;

}
