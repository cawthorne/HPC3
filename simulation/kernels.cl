#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9

/* struct to hold the 'speed' values */
typedef struct {
    float speeds[NSPEEDS];
} speed_t;

/* struct to hold the parameter values */
typedef struct {
    int nx;            /* no. of cells in x-direction */
    int ny;            /* no. of cells in y-direction */
    int max_iters;      /* no. of iterations */
    int reynolds_dim;  /* dimension for Reynolds number */
    float density;       /* density per link */
    float accel;         /* density redistribution */
    float omega;         /* relaxation parameter */
} param_t;

typedef enum { ACCEL_ROW=0, ACCEL_COLUMN=1 } accel_e;
typedef struct {
    int col_or_row;
    int idx;
} accel_area_t;

/*
*   TODO
*   Write OpenCL kernels
*/

float total_density(const param_t params, __global float* cells)
{
    int ii,jj,kk;        /* generic counters */
    float total = 0.0;  /* accumulator */

    for (ii = 0; ii < params.ny; ii++)
    {
        for (jj = 0; jj < params.ny; jj++)
        {
            for (kk = 0; kk < NSPEEDS; kk++)
            {
                total += cells[(ii*params.nx + jj)*9 + kk];
            }
        }
    }

    return total;
}


void reduce(                                          
   __local  float*    local_sums,                          
   __global float*    partial_sums)                        
{                                                          
   int num_wrk_items  = get_local_size(0)*get_local_size(1);                 
   int local_id       = get_local_id(0)*get_local_size(1) + get_local_id(1);                   
   int group_id       = get_group_id(0)*get_num_groups(1) + get_group_id(1);                   
   
   float sum;  
   int tots;
   int i;                                      
   
   if (local_id == 0) {                      
      sum = 0.0f;                            
   
      for (i=0; i<num_wrk_items; i++) {        
          sum += local_sums[i];             
      }                                     
   
      partial_sums[group_id] = sum;         
   }
}

__kernel void acc_flow(param_t params, accel_area_t accel_area,  __global float* cells, __global int* obstacles){
	
	 float w1,w2;  /* weighting factors */
	int jj,ii;
    /* compute weighting factors */
    w1 = params.density * params.accel / 9.0;
    w2 = params.density * params.accel / 36.0;

    if (accel_area.col_or_row == ACCEL_COLUMN)
    {
        jj = accel_area.idx;
			
			ii = get_global_id(0);

			int current_index = (ii*params.nx + jj)*9;
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
            (cells[ii*params.nx + jj + 4] - w1) > 0.0 &&
            (cells[ii*params.nx + jj + 7] - w2) > 0.0 &&
            (cells[ii*params.nx + jj + 8] - w2) > 0.0 )
            {
               /* increase 'north-side' densities */
                cells[current_index + 2] += w1;
                cells[current_index + 5] += w2;
                cells[current_index + 6] += w2;
                /* decrease 'south-side' densities */
                cells[current_index + 4] -= w1;
                cells[current_index + 7] -= w2;
                cells[current_index + 8] -= w2;
				
            }
    }
    else
    {
        ii = accel_area.idx;

			jj = get_global_id(0);
			
			int current_index = (ii*params.nx + jj)*9;
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
            (cells[current_index + 3] - w1) > 0.0 &&
            (cells[current_index + 6] - w2) > 0.0 &&
            (cells[current_index + 7] - w2) > 0.0 )
            {
				/* increase 'east-side' densities */
                cells[current_index + 1] += w1;
                cells[current_index + 5] += w2;
                cells[current_index + 8] += w2;
                /* decrease 'west-side' densities */
                cells[current_index + 3] -= w1;
                cells[current_index + 6] -= w2;
                cells[current_index + 7] -= w2;
            }
    }
}


__kernel void propagate(param_t params, accel_area_t accel_area, __global float* cells, __global float* tmp_cells, __global int* obstacles, __local float* l_us, __global float* g_us){


int kk;       /* generic counters */
         
    const float w0 = 4.0/9.0;    /* weighting factor */
    const float w1C = 1.0/9.0;    /* weighting factor */
    const float w2C = 1.0/36.0;   /* weighting factor */

    float u_x,u_y;               /* av. velocities in x and y directions */
    float u_sq;                  /* squared velocity */
    float local_density;         /* sum of densities in a particular cell */
    float u[NSPEEDS];            /* directional velocities */
    float d_equ[NSPEEDS];        /* equilibrium densities */

	int x_e,x_w,y_n,y_s;  /* indices of neighbouring cells */

    float w1,w2;  
				int current_local_id = get_local_id(1)*get_local_size(0) + get_local_id(0);
				l_us[current_local_id] = 0;
	
int jj,ii;
    /* compute weighting factors */
    w1 = params.density * params.accel / 9.0;
    w2 = params.density * params.accel / 36.0;

	ii = get_global_id(0);
	jj = get_global_id(1);

			if (ii + 1 == params.ny){
				y_n = 0;
			} else {
				y_n = ii + 1;
			}
			if (jj + 1 == params.nx){
				x_e = 0;
			} else {
				x_e = jj + 1;
			}
            y_s = (ii == 0) ? (params.ny - 1) : (ii - 1);
            x_w = (jj == 0) ? (params.nx - 1) : (jj - 1);
            /* propagate densities to neighbouring cells, following
            ** appropriate directions of travel and writing into
            ** scratch space grid */

			int current_index = (ii*params.nx + jj)*9;
				tmp_cells[current_index] = cells[current_index];
				tmp_cells[current_index + 1] = cells[(ii*params.nx + x_w)*9 + 1];		
				tmp_cells[current_index + 2] = cells[(y_s*params.nx + jj)*9  + 2];				
				tmp_cells[current_index + 3] = cells[(ii*params.nx + x_e)*9 + 3];
				tmp_cells[current_index + 4] = cells[(y_n*params.nx + jj)*9 + 4];  
				tmp_cells[current_index + 5] = cells[(y_s*params.nx + x_w)*9  + 5]; 
				tmp_cells[current_index + 6] = cells[(y_s*params.nx + x_e)*9  + 6];
				tmp_cells[current_index + 7] = cells[(y_n*params.nx + x_e)*9 + 7]; 
				tmp_cells[current_index + 8] = cells[(y_n*params.nx + x_w)*9  + 8];
				tmp_cells[current_index + 5] = cells[(y_s*params.nx + x_w)*9  + 5]; 	

			if (!obstacles[ii*params.nx + jj])
            {

                /* compute local density total */
                local_density = 0.0;
				
				
                for (kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += tmp_cells[current_index + kk];
                }



                /* compute x velocity component */
                u_x = (tmp_cells[current_index + 1] +
                        tmp_cells[current_index + 5] +
                        tmp_cells[current_index + 8]
                    - (tmp_cells[current_index + 3] +
                        tmp_cells[current_index + 6] +
                        tmp_cells[current_index + 7]))
                    / local_density;

                /* compute y velocity component */
                u_y = (tmp_cells[current_index + 2] +
                        tmp_cells[current_index + 5] +
                        tmp_cells[current_index + 6]
                    - (tmp_cells[current_index + 4] +
                        tmp_cells[current_index + 7] +
                        tmp_cells[current_index + 8]))
                    / local_density;
	
                /* velocity squared */
                u_sq = u_x * u_x + u_y * u_y;

                /* directional velocity components */
                u[1] =   u_x;        /* east */
                u[2] =         u_y;  /* north */
                u[3] = - u_x;        /* west */
                u[4] =       - u_y;  /* south */
                u[5] =   u_x + u_y;  /* north-east */
                u[6] = - u_x + u_y;  /* north-west */
                u[7] = - u_x - u_y;  /* south-west */
                u[8] =   u_x - u_y;  /* south-east */

                /* equilibrium densities */
                /* zero velocity density: weight w0 */
                d_equ[0] = w0 * local_density * (1.0 - u_sq * 1.5);
                /* axis speeds: weight w1 */
                d_equ[1] = w1C * local_density * (1.0 + u[1] * 3
                    + (u[1] * u[1]) * (4.5)
                    - u_sq * 1.5);
                d_equ[2] = w1C * local_density * (1.0 + u[2] * 3
                    + (u[2] * u[2]) * (4.5)
                    - u_sq * 1.5);
                d_equ[3] = w1C * local_density * (1.0 + u[3] * 3
                    + (u[3] * u[3]) * (4.5)
                    - u_sq * 1.5);
                d_equ[4] = w1C * local_density * (1.0 + u[4] * 3
                    + (u[4] * u[4]) * (4.5)
                    - u_sq * 1.5);
                /* diagonal speeds: weight w2 */
                d_equ[5] = w2C * local_density * (1.0 + u[5] * 3
                    + (u[5] * u[5]) * (4.5)
                    - u_sq * 1.5);
                d_equ[6] = w2C * local_density * (1.0 + u[6] * 3
                    + (u[6] * u[6]) * (4.5)
                    - u_sq * 1.5);
                d_equ[7] = w2C * local_density * (1.0 + u[7] * 3
                    + (u[7] * u[7]) * (4.5)
                    - u_sq * 1.5);
                d_equ[8] = w2C * local_density * (1.0 + u[8] * 3
                    + (u[8] * u[8]) * (4.5)
                    - u_sq * 1.5);

                /* relaxation step */
                for (kk = 0; kk < NSPEEDS; kk++)
                {
                    tmp_cells[current_index + kk] = 
                        (tmp_cells[current_index + kk] + params.omega * 
                        (d_equ[kk] - tmp_cells[current_index + kk]));
					
				
                }
				
				//printf("\nquack1 %f\n",local_density);
				/* local density total */
                local_density = 0.0;

                for (kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += tmp_cells[current_index + kk];
                }

                /* x-component of velocity */
                u_x = (tmp_cells[current_index + 1] +
                        tmp_cells[current_index + 5] +
                        tmp_cells[current_index + 8]
                    - (tmp_cells[current_index + 3] +
                        tmp_cells[current_index + 6] +
                        tmp_cells[current_index + 7])) /
                    local_density;

                /* compute y velocity component */
                u_y = (tmp_cells[current_index + 2] +
                        tmp_cells[current_index + 5] +
                        tmp_cells[current_index + 6]
                    - (tmp_cells[current_index + 4] +
                        tmp_cells[current_index + 7] +
                        tmp_cells[current_index + 8])) /
                    local_density;
				
                /* accumulate the norm of x- and y- velocity components */
                l_us[current_local_id] = sqrt(u_x*u_x + u_y*u_y);
				
				//printf("\nquack2 %f\n",local_density);
				if (l_us[current_local_id] != 0){
					//printf("not zero");
				}
				
            } else {
				
					float v1 = tmp_cells[current_index + 3];
					float v2 = tmp_cells[current_index + 4];
					tmp_cells[current_index + 3] = tmp_cells[current_index + 1];
					tmp_cells[current_index + 4] = tmp_cells[current_index + 2];
					float v5 = tmp_cells[current_index + 7];
					float v6 = tmp_cells[current_index + 8];
					tmp_cells[current_index + 7] = tmp_cells[current_index + 5];
					tmp_cells[current_index + 8] = tmp_cells[current_index + 6];
					tmp_cells[current_index + 1] = v1;
					tmp_cells[current_index + 2] = v2;
					tmp_cells[current_index + 5] = v5;
					tmp_cells[current_index + 6] = v6;
			}
		 barrier(CLK_LOCAL_MEM_FENCE);
		reduce(l_us, g_us);
		
		//if (get_global_id(0) == 0 && get_global_id(1) == 0){
		//printf("tot density: %.12E\n", total_density(params, tmp_cells));}
		
		//barrier(CLK_GLOBAL_MEM_FENCE);
		
}