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


void reduceL(                                          
   __local  float*    local_sums,                          
   __global float*    partial_sums, int iter)                        
{                                                          
   int num_wrk_items  = get_local_size(0)*get_local_size(1);                 
   int local_id       = get_local_id(1)*get_local_size(0) + get_local_id(0);                   
   int group_id       = get_num_groups(0)*get_num_groups(1)*iter + get_group_id(0)*get_num_groups(1) + get_group_id(1);                   
                                      
   int mask;
   for (mask=num_wrk_items;mask>1;){
   if ((mask & 1) == 1 && local_id == 0){
		local_sums[0] += local_sums[mask-1];
   }
   mask=mask>>1;

	if (local_id < mask) {  
 
	  local_sums[local_id] += local_sums[local_id+mask];
	  
	}

	barrier(CLK_LOCAL_MEM_FENCE);
   }
   partial_sums[(long)group_id] = local_sums[0]; 
}

void reduce(                                          
   __local  float*    local_sums,                          
   __global float*    partial_sums, int iter)                        
{                                                          
   int num_wrk_items  = get_local_size(0)*get_local_size(1);                 
   int local_id       = get_local_id(1)*get_local_size(0) + get_local_id(0);                   
   int group_id       = get_num_groups(0)*get_num_groups(1)*iter + get_group_id(0)*get_num_groups(1) + get_group_id(1);                   
                                      
   int mask;
   for (mask=num_wrk_items;mask>1;){
   
   mask=mask>>1;

	if (local_id < mask) {  
 
	  local_sums[local_id] += local_sums[local_id+mask];
	  
	}

	barrier(CLK_LOCAL_MEM_FENCE);
   }
   partial_sums[group_id] = local_sums[0]; 
}


__kernel void propagate(param_t params, accel_area_t accel_area, __global float* cells, __global float* tmp_cells, __global int* obstacles, __local float* l_us, __global float* g_us, const int iter){


     int ii,jj;     /* generic counters */
	

int kk;       /* generic counters */
         
    const float w0C = 4.0/9.0;    /* weighting factor */
    const float w1C = 1.0/9.0;    /* weighting factor */
    const float w2C = 1.0/36.0;   /* weighting factor */


	int x_e,x_w,y_n,y_s;  /* indices of neighbouring cells */

				int current_local_id = get_local_id(1)*get_local_size(0) + get_local_id(0);
				l_us[current_local_id] = 0;
	
    /* compute weighting factors */

    float u_x,u_y;               /* av. velocities in x and y directions */
    float u_sq;                  /* squtmp1red velocity */
    float local_density;         /* sum of densities in a particular cell */
	
	ii = get_global_id(1);
	jj = get_global_id(0);

			y_n = (ii + 1 == params.ny) ? (0) : (ii + 1);
            x_e = (jj + 1 == params.nx) ? (0) : (jj + 1);
            y_s = (ii == 0) ? (params.ny - 1) : (ii - 1);
            x_w = (jj == 0) ? (params.nx - 1) : (jj - 1);
            /* propagate densities to neighbouring cells, following
            ** appropriate directions of travel and writing into
            ** scratch space grid */
			
			 float local_cell[NSPEEDS];
				
			int current_index = (ii*params.nx + jj);
			
				local_cell[0] = cells[current_index];
				local_cell[1] = cells[(ii*params.nx + x_w) + params.nx*params.ny*1];		
				local_cell[2] = cells[(y_s*params.nx + jj) + params.nx*params.ny*2];				
				local_cell[3] = cells[(ii*params.nx + x_e) + params.nx*params.ny*3];
				local_cell[4] = cells[(y_n*params.nx + jj) + params.nx*params.ny*4];  
				local_cell[5] = cells[(y_s*params.nx + x_w) + params.nx*params.ny*5]; 
				local_cell[6] = cells[(y_s*params.nx + x_e) + params.nx*params.ny*6];
				local_cell[7] = cells[(y_n*params.nx + x_e) + params.nx*params.ny*7]; 
				local_cell[8] = cells[(y_n*params.nx + x_w) + params.nx*params.ny*8];	


            	local_density = local_cell[0] 
				+local_cell[1]+local_cell[2]+ 
				local_cell[3]+local_cell[4]+local_cell[5]+ 
				local_cell[6]+local_cell[7]+local_cell[8];
				
                /* compute x velocity component */
                u_x = (local_cell[1] + local_cell[5] + local_cell[8] - (local_cell[3] + 
				local_cell[6] + local_cell[7]))/local_density;

                /* compute y velocity component */
                u_y = (local_cell[2] + local_cell[5] + local_cell[6]- (local_cell[4] + 
				local_cell[7] 
				+ local_cell[8]))/ local_density;

                /* velocity squtmp1red */
                u_sq = u_x * u_x + u_y * u_y;
                
				const float utmp1 =   u_x + u_y; 

                const float utmp2 = - u_x + u_y;

				const int b = obstacles[ii*params.nx + jj];
				
		
				tmp_cells[current_index + params.nx*params.ny*0] = b ? local_cell[0] : (local_cell[0] + params.omega * ( w0C * local_density * (1.0 - u_sq * (1.5)) - local_cell[0]));
                
				tmp_cells[current_index + params.nx*params.ny*1] = b ? local_cell[3] : (local_cell[1] + params.omega * ( w1C * local_density * (1.0 + u_x * 3.0
                    + (u_x * u_x)*4.5
                    - u_sq *(1.5)) - local_cell[1]));
				tmp_cells[current_index + params.nx*params.ny*2] = b ? local_cell[4]:(local_cell[2] + params.omega * (w1C * local_density * (1.0 + u_y * 3.0
                    + (u_y * u_y)*4.5
                    - u_sq *(1.5)) - local_cell[2]));
				tmp_cells[current_index + params.nx*params.ny*3] = b ? local_cell[1]:(local_cell[3] + params.omega * (w1C * local_density * (1.0 - u_x* 3.0
                    + (u_x * u_x)*4.5
                    - u_sq *(1.5)) - local_cell[3]));
				tmp_cells[current_index + params.nx*params.ny*4] = b ? local_cell[2]:(local_cell[4] + params.omega * ( w1C * local_density * (1.0 - u_y * 3.0
                    + (u_y * u_y)*4.5
                    - u_sq *(1.5)) - local_cell[4]));
				tmp_cells[current_index + params.nx*params.ny*5] = b ? local_cell[7]:(local_cell[5] + params.omega * (w2C * local_density * (1.0 + utmp1 * 3.0
                    + (utmp1 * utmp1)*4.5
                    - u_sq *(1.5)) - local_cell[5]));
				tmp_cells[current_index + params.nx*params.ny*6] = b ? local_cell[8]:(local_cell[6] + params.omega * (w2C * local_density * (1.0 + utmp2 * 3.0
                    + (utmp2 * utmp2)*4.5
                    - u_sq *(1.5)) - local_cell[6]));
				tmp_cells[current_index + params.nx*params.ny*7] = b ? local_cell[5]:(local_cell[7] + params.omega * (w2C * local_density * (1.0 - utmp1 * 3.0
                    + (utmp1 * utmp1)*4.5
                    - u_sq *(1.5)) - local_cell[7]));
				tmp_cells[current_index + params.nx*params.ny*8] = b ? local_cell[6]:(local_cell[8] + params.omega * (w2C * local_density * (1.0 - utmp2 * 3.0
                    + (utmp2 * utmp2)*4.5
                    - u_sq *(1.5)) - local_cell[8]));
					
				local_density =0.0f;
					

					
                local_density = (tmp_cells[current_index + params.nx*params.ny*0] +
							 tmp_cells[current_index + params.nx*params.ny*1]+
							 tmp_cells[current_index + params.nx*params.ny*2]+
							 tmp_cells[current_index + params.nx*params.ny*3]+
							 tmp_cells[current_index + params.nx*params.ny*4]+
							 tmp_cells[current_index + params.nx*params.ny*5]+
							 tmp_cells[current_index + params.nx*params.ny*6]+
							 tmp_cells[current_index + params.nx*params.ny*7]+
							 tmp_cells[current_index + params.nx*params.ny*8]);

                /* x-component of velocity */
                u_x = b ? u_x : 
				(tmp_cells[current_index + params.nx*params.ny*1] +
                        tmp_cells[current_index + params.nx*params.ny*5] +
                        tmp_cells[current_index + params.nx*params.ny*8]
                    - (tmp_cells[current_index + params.nx*params.ny*3] +
                        tmp_cells[current_index + params.nx*params.ny*6] +
                        tmp_cells[current_index + params.nx*params.ny*7])) /
                    local_density;

                /* compute y velocity component */
                u_y =  b ? u_y : 
				(tmp_cells[current_index + params.nx*params.ny*2] +
                        tmp_cells[current_index + params.nx*params.ny*5] +
                        tmp_cells[current_index + params.nx*params.ny*6]
                    - (tmp_cells[current_index + params.nx*params.ny*4] +
                        tmp_cells[current_index + params.nx*params.ny*7] +
                        tmp_cells[current_index + params.nx*params.ny*8])) /
                    local_density;
				
                /* accumulate the norm of x- and y- velocity components */
                l_us[current_local_id] = b ? 0 : sqrt(u_x*u_x + u_y*u_y);
			

		 barrier(CLK_LOCAL_MEM_FENCE);
		
		 reduce(l_us, g_us, iter);
	
	    float w1,w2;  /* weighting factors */

    /* compute weighting factors */
    w1 = params.density * params.accel * w1C;
    w2 = params.density * params.accel * w2C;
	
    if (accel_area.col_or_row == ACCEL_COLUMN && get_global_id(0) == 3)
    {
        jj = accel_area.idx;

        ii= get_global_id(1);
        
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
            (tmp_cells[ii*params.nx + jj + params.nx*params.ny*4] - w1) > 0.0 &&
            (tmp_cells[ii*params.nx + jj + params.nx*params.ny*7] - w2) > 0.0 &&
            (tmp_cells[ii*params.nx + jj + params.nx*params.ny*8] - w2) > 0.0 )
            {
                /* increase 'north-side' densities */
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*2] += w1;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*5] += w2;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*6] += w2;
                /* decrease 'south-side' densities */
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*4] -= w1;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*7] -= w2;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*8] -= w2;
            }
        
    }
    else if (accel_area.col_or_row != ACCEL_COLUMN && get_global_id(1) == 3)
    {
        ii = accel_area.idx;

        jj = get_global_id(0);
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
            (tmp_cells[ii*params.nx + jj + params.nx*params.ny*3] - w1) > 0.0 &&
            (tmp_cells[ii*params.nx + jj + params.nx*params.ny*6] - w2) > 0.0 &&
            (tmp_cells[ii*params.nx + jj + params.nx*params.ny*7] - w2) > 0.0 )
            {
                /* increase 'east-side' densities */
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*1] += w1;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*5] += w2;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*8] += w2;
                /* decrease 'west-side' densities */
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*3] -= w1;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*6] -= w2;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*7] -= w2;
            }

    }

}

__kernel void propagateNotPow2(param_t params, accel_area_t accel_area, __global float* cells, __global float* tmp_cells, __global int* obstacles, __local float* l_us, __global float* g_us, const int iter){



     int ii,jj;     /* generic counters */
	

int kk;       /* generic counters */
         
    const float w0C = 4.0/9.0;    /* weighting factor */
    const float w1C = 1.0/9.0;    /* weighting factor */
    const float w2C = 1.0/36.0;   /* weighting factor */


	int x_e,x_w,y_n,y_s;  /* indices of neighbouring cells */

				int current_local_id = get_local_id(1)*get_local_size(0) + get_local_id(0);
				l_us[current_local_id] = 0;
	
    /* compute weighting factors */

    float u_x,u_y;               /* av. velocities in x and y directions */
    float u_sq;                  /* squtmp1red velocity */
    float local_density;         /* sum of densities in a particular cell */
	
	ii = get_global_id(1);
	jj = get_global_id(0);


			y_n = (ii + 1 == params.ny) ? (0) : (ii + 1);
            x_e = (jj + 1 == params.nx) ? (0) : (jj + 1);
            y_s = (ii == 0) ? (params.ny - 1) : (ii - 1);
            x_w = (jj == 0) ? (params.nx - 1) : (jj - 1);
            /* propagate densities to neighbouring cells, following
            ** appropriate directions of travel and writing into
            ** scratch space grid */
			
			 float local_cell[NSPEEDS];
				
			int current_index = (ii*params.nx + jj);
			
				local_cell[0] = cells[current_index];
				local_cell[1] = cells[(ii*params.nx + x_w) + params.nx*params.ny*1];		
				local_cell[2] = cells[(y_s*params.nx + jj) + params.nx*params.ny*2];				
				local_cell[3] = cells[(ii*params.nx + x_e) + params.nx*params.ny*3];
				local_cell[4] = cells[(y_n*params.nx + jj) + params.nx*params.ny*4];  
				local_cell[5] = cells[(y_s*params.nx + x_w) + params.nx*params.ny*5]; 
				local_cell[6] = cells[(y_s*params.nx + x_e) + params.nx*params.ny*6];
				local_cell[7] = cells[(y_n*params.nx + x_e) + params.nx*params.ny*7]; 
				local_cell[8] = cells[(y_n*params.nx + x_w) + params.nx*params.ny*8];	


            	local_density = local_cell[0] 
				+local_cell[1]+local_cell[2]+ 
				local_cell[3]+local_cell[4]+local_cell[5]+ 
				local_cell[6]+local_cell[7]+local_cell[8];
				
                /* compute x velocity component */
                u_x = (local_cell[1] + local_cell[5] + local_cell[8] - (local_cell[3] + 
				local_cell[6] + local_cell[7]))/local_density;

                /* compute y velocity component */
                u_y = (local_cell[2] + local_cell[5] + local_cell[6]- (local_cell[4] + 
				local_cell[7] 
				+ local_cell[8]))/ local_density;

                /* velocity squtmp1red */
                u_sq = u_x * u_x + u_y * u_y;
                
				const float utmp1 =   u_x + u_y; 

                const float utmp2 = - u_x + u_y;

				const int b = obstacles[ii*params.nx + jj];
				
		
				tmp_cells[current_index + params.nx*params.ny*0] = b ? local_cell[0] : (local_cell[0] + params.omega * ( w0C * local_density * (1.0 - u_sq * (1.5)) - local_cell[0]));
                
				tmp_cells[current_index + params.nx*params.ny*1] = b ? local_cell[3] : (local_cell[1] + params.omega * ( w1C * local_density * (1.0 + u_x * 3.0
                    + (u_x * u_x)*4.5
                    - u_sq *(1.5)) - local_cell[1]));
				tmp_cells[current_index + params.nx*params.ny*2] = b ? local_cell[4]:(local_cell[2] + params.omega * (w1C * local_density * (1.0 + u_y * 3.0
                    + (u_y * u_y)*4.5
                    - u_sq *(1.5)) - local_cell[2]));
				tmp_cells[current_index + params.nx*params.ny*3] = b ? local_cell[1]:(local_cell[3] + params.omega * (w1C * local_density * (1.0 - u_x* 3.0
                    + (u_x * u_x)*4.5
                    - u_sq *(1.5)) - local_cell[3]));
				tmp_cells[current_index + params.nx*params.ny*4] = b ? local_cell[2]:(local_cell[4] + params.omega * ( w1C * local_density * (1.0 - u_y * 3.0
                    + (u_y * u_y)*4.5
                    - u_sq *(1.5)) - local_cell[4]));
				tmp_cells[current_index + params.nx*params.ny*5] = b ? local_cell[7]:(local_cell[5] + params.omega * (w2C * local_density * (1.0 + utmp1 * 3.0
                    + (utmp1 * utmp1)*4.5
                    - u_sq *(1.5)) - local_cell[5]));
				tmp_cells[current_index + params.nx*params.ny*6] = b ? local_cell[8]:(local_cell[6] + params.omega * (w2C * local_density * (1.0 + utmp2 * 3.0
                    + (utmp2 * utmp2)*4.5
                    - u_sq *(1.5)) - local_cell[6]));
				tmp_cells[current_index + params.nx*params.ny*7] = b ? local_cell[5]:(local_cell[7] + params.omega * (w2C * local_density * (1.0 - utmp1 * 3.0
                    + (utmp1 * utmp1)*4.5
                    - u_sq *(1.5)) - local_cell[7]));
				tmp_cells[current_index + params.nx*params.ny*8] = b ? local_cell[6]:(local_cell[8] + params.omega * (w2C * local_density * (1.0 - utmp2 * 3.0
                    + (utmp2 * utmp2)*4.5
                    - u_sq *(1.5)) - local_cell[8]));
					
				local_density =0.0f;
					

					
                local_density = (tmp_cells[current_index + params.nx*params.ny*0] +
							 tmp_cells[current_index + params.nx*params.ny*1]+
							 tmp_cells[current_index + params.nx*params.ny*2]+
							 tmp_cells[current_index + params.nx*params.ny*3]+
							 tmp_cells[current_index + params.nx*params.ny*4]+
							 tmp_cells[current_index + params.nx*params.ny*5]+
							 tmp_cells[current_index + params.nx*params.ny*6]+
							 tmp_cells[current_index + params.nx*params.ny*7]+
							 tmp_cells[current_index + params.nx*params.ny*8]);

                /* x-component of velocity */
                u_x = b ? u_x : 
				(tmp_cells[current_index + params.nx*params.ny*1] +
                        tmp_cells[current_index + params.nx*params.ny*5] +
                        tmp_cells[current_index + params.nx*params.ny*8]
                    - (tmp_cells[current_index + params.nx*params.ny*3] +
                        tmp_cells[current_index + params.nx*params.ny*6] +
                        tmp_cells[current_index + params.nx*params.ny*7])) /
                    local_density;

                /* compute y velocity component */
                u_y =  b ? u_y : 
				(tmp_cells[current_index + params.nx*params.ny*2] +
                        tmp_cells[current_index + params.nx*params.ny*5] +
                        tmp_cells[current_index + params.nx*params.ny*6]
                    - (tmp_cells[current_index + params.nx*params.ny*4] +
                        tmp_cells[current_index + params.nx*params.ny*7] +
                        tmp_cells[current_index + params.nx*params.ny*8])) /
                    local_density;
				
                /* accumulate the norm of x- and y- velocity components */
                l_us[current_local_id] = b ? 0 : sqrt(u_x*u_x + u_y*u_y);
			

		 barrier(CLK_LOCAL_MEM_FENCE);
		
		 reduceL(l_us, g_us, iter);
	
	    float w1,w2;  /* weighting factors */

    /* compute weighting factors */
    w1 = params.density * params.accel * w1C;
    w2 = params.density * params.accel * w2C;

    if (accel_area.col_or_row == ACCEL_COLUMN && get_global_id(0) == 3)
    {
        jj = accel_area.idx;

        ii= get_global_id(1);
        
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
            (tmp_cells[ii*params.nx + jj + params.nx*params.ny*4] - w1) > 0.0 &&
            (tmp_cells[ii*params.nx + jj + params.nx*params.ny*7] - w2) > 0.0 &&
            (tmp_cells[ii*params.nx + jj + params.nx*params.ny*8] - w2) > 0.0 )
            {
                /* increase 'north-side' densities */
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*2] += w1;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*5] += w2;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*6] += w2;
                /* decrease 'south-side' densities */
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*4] -= w1;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*7] -= w2;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*8] -= w2;
            }
        
    }
    else if (accel_area.col_or_row != ACCEL_COLUMN && get_global_id(1) == 3)
    {
        ii = accel_area.idx;

        jj = get_global_id(0);
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
            (tmp_cells[ii*params.nx + jj + params.nx*params.ny*3] - w1) > 0.0 &&
            (tmp_cells[ii*params.nx + jj + params.nx*params.ny*6] - w2) > 0.0 &&
            (tmp_cells[ii*params.nx + jj + params.nx*params.ny*7] - w2) > 0.0 )
            {
                /* increase 'east-side' densities */
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*1] += w1;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*5] += w2;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*8] += w2;
                /* decrease 'west-side' densities */
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*3] -= w1;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*6] -= w2;
                tmp_cells[ii*params.nx + jj + params.nx*params.ny*7] -= w2;
            }

    }
}