__kernel void mandelbrot (__global double *real_input, __global double *imaginary_input,  __global int *mandelbrot, __global int *max_iterations)
{
	int idx = get_global_id(0);
	double r_input = real_input[idx];
	double i_input = imaginary_input[idx];

	double abs_z_value = 0;
	int counter = 0;	

	double r_input_2 = 0;
	double i_input_2 = 0;


	int max_iter = *max_iterations;

	for (int n=0 ; n < max_iter ; n++)
	{
		counter++;

		r_input_2 = r_input;
		i_input_2 = i_input;

		abs_z_value = (r_input_2 * r_input_2 ) + (i_input * i_input);
		
		if (abs_z_value > 4)
		{
			break;
		}

		r_input = (r_input_2 * r_input_2) - (i_input_2 * i_input_2);
		i_input = 2 * r_input_2 * i_input_2;
	}
				 
	mandelbrot[idx] = counter;

}
