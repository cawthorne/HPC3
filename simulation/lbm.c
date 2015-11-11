/*
** code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the bhatnagar-gross-krook collision step.
**
** the 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** a 2d grid:
**
**           cols
**       --- --- ---
**      | d | e | f |
** rows  --- --- ---
**      | a | b | c |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1d array:
**
**  --- --- --- --- --- ---
** | a | b | c | d | e | f |
**  --- --- --- --- --- ---
**
** grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./lbm -a av_vels.dat -f final_state.dat -p ../inputs/box.params
**
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

//#define DEBUG

#include "lbm.h"

char* printErr(cl_int err){
	 switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default: return "Unknown";
    }
}

void accelerate_flow(const param_t params, const accel_area_t accel_area,
    float* cells, int* obstacles)
{
    int ii,jj;     /* generic counters */
    double w1,w2;  /* weighting factors */

    /* compute weighting factors */
    w1 = params.density * params.accel / 9.0;
    w2 = params.density * params.accel / 36.0;

    if (accel_area.col_or_row == ACCEL_COLUMN)
    {
        jj = accel_area.idx;

        for (ii = 0; ii < params.ny; ii++)
        {
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
            (cells[(ii*params.nx + jj)*9 + 4] - w1) > 0.0 &&
            (cells[(ii*params.nx + jj)*9 + 7] - w2) > 0.0 &&
            (cells[(ii*params.nx + jj)*9 + 8] - w2) > 0.0 )
            {
                /* increase 'north-side' densities */
                cells[(ii*params.nx + jj)*9 + 2] += w1;
                cells[(ii*params.nx + jj)*9 + 5] += w2;
                cells[(ii*params.nx + jj)*9 + 6] += w2;
                /* decrease 'south-side' densities */
                cells[(ii*params.nx + jj)*9 + 4] -= w1;
                cells[(ii*params.nx + jj)*9 + 7] -= w2;
                cells[(ii*params.nx + jj)*9 + 8] -= w2;
            }
        }
    }
    else
    {
        ii = accel_area.idx;

        for (jj = 0; jj < params.nx; jj++)
        {
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
            (cells[(ii*params.nx + jj)*9 + 3] - w1) > 0.0 &&
            (cells[(ii*params.nx + jj)*9 + 6] - w2) > 0.0 &&
            (cells[(ii*params.nx + jj)*9 + 7] - w2) > 0.0 )
            {
                /* increase 'east-side' densities */
                cells[(ii*params.nx + jj)*9 + 1] += w1;
                cells[(ii*params.nx + jj)*9 + 5] += w2;
                cells[(ii*params.nx + jj)*9 + 8] += w2;
                /* decrease 'west-side' densities */
                cells[(ii*params.nx + jj)*9 + 3] -= w1;
                cells[(ii*params.nx + jj)*9 + 6] -= w2;
                cells[(ii*params.nx + jj)*9 + 7] -= w2;
            }
        }
    }
}

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
    char * final_state_file = NULL;
    char * av_vels_file = NULL;
    char * param_file = NULL;

    accel_area_t accel_area;

    param_t  params;              /* struct to hold parameter values */
    float* cells     = NULL;    /* grid containing fluid densities */
    float* tmp_cells = NULL;    /* scratch space */
    int*     obstacles = NULL;    /* grid indicating which cells are blocked */
    float*  av_vels   = NULL;    /* a record of the av. velocity computed for each timestep */

                        /*  generic counter */
    struct timeval timstr;        /* structure to hold elapsed time */
    struct rusage ru;             /* structure to hold CPU time--system and user */
    double tic,toc;               /* floating point numbers to calculate elapsed wallclock time */
    float usrtim;                /* floating point number to record elapsed user CPU time */
    float systim;                /* floating point number to record elapsed system CPU time */

    int device_id;
    lbm_context_t lbm_context;

    parse_args(argc, argv, &final_state_file, &av_vels_file, &param_file, &device_id);

    initialise(param_file, &accel_area, &params, &cells, &tmp_cells, &obstacles, &av_vels);
    opencl_initialise(device_id, params, accel_area, &lbm_context, cells, obstacles, tmp_cells);

	float* us = malloc(sizeof(float)*params.max_iters*params.nx*params.ny/(lbm_context.local_size*lbm_context.local_size));

	int ob_num = 0;
	int i;
	for (i=0;i<params.nx*params.ny;i++){
		ob_num += !obstacles[i];
	}
	
	size_t global[2] = {params.nx,params.ny};
	
	size_t local[2] = {lbm_context.local_size,lbm_context.local_size};
	
    /* iterate for max_iters timesteps */
    gettimeofday(&timstr,NULL);
    tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);
	int iiM;
	cl_int err;

    for (iiM = 0; iiM < params.max_iters; iiM++)
    {
	err = clSetKernelArg(lbm_context.kernel[0], 7, sizeof(int), &iiM);

	err |= clEnqueueNDRangeKernel(lbm_context.queue, lbm_context.kernel[0], 2, NULL, global, local, 0, NULL, NULL);
	
	
	if ((iiM & 1) == 0){
		err |= clSetKernelArg(lbm_context.kernel[0], 2, sizeof(cl_mem), &lbm_context.args[3]);
		err |= clSetKernelArg(lbm_context.kernel[0], 3, sizeof(cl_mem), &lbm_context.args[2]);

	} else {

		err |= clSetKernelArg(lbm_context.kernel[0], 2, sizeof(cl_mem), &lbm_context.args[2]);
		err |= clSetKernelArg(lbm_context.kernel[0], 3, sizeof(cl_mem), &lbm_context.args[3]);
	}
	

    }
	
	err |= clEnqueueReadBuffer(lbm_context.queue, lbm_context.args[6], CL_TRUE,
         0, sizeof(float)*params.max_iters*params.nx*params.ny/(lbm_context.local_size*lbm_context.local_size), us, 0, NULL, NULL);
	
	
	if ((iiM & 1) == 0){
		err = clEnqueueReadBuffer(lbm_context.queue, lbm_context.args[2], CL_TRUE,
         0, sizeof(float)*params.nx*params.ny*9, cells, 0, NULL, NULL);

	} else {
		err = clEnqueueReadBuffer(lbm_context.queue, lbm_context.args[3], CL_TRUE,
         0, sizeof(float)*params.nx*params.ny*9, cells, 0, NULL, NULL);
	}
		if (CL_SUCCESS != err){printErr(err); DIE("OpenCL error %d", err);}
		
	int j;
	float u;
	for (i = 0;i<params.max_iters;i++){
		u=0;
		for (j = 0;j<params.nx*params.ny/(lbm_context.local_size*lbm_context.local_size);j++){
			u += us[i*params.nx*params.ny/(lbm_context.local_size*lbm_context.local_size) + j];
		}
		av_vels[i] = u/(float)ob_num;
	    //printf("\navs %.12E %d %.12E %.12E\n",u,ob_num,calc_reynolds(params,av_vels[i]), total_density(params, cells));
	}

    // Do not remove this, or the timing will be incorrect!
    clFinish(lbm_context.queue);

    gettimeofday(&timstr,NULL);
    toc=timstr.tv_sec+(timstr.tv_usec/1000000.0);
    getrusage(RUSAGE_SELF, &ru);
    timstr=ru.ru_utime;
    usrtim=timstr.tv_sec+(timstr.tv_usec/1000000.0);
    timstr=ru.ru_stime;
    systim=timstr.tv_sec+(timstr.tv_usec/1000000.0);

    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params,av_vels[params.max_iters-1]));
    printf("Elapsed time:\t\t\t%.6f (s)\n", toc-tic);
    printf("Elapsed user CPU time:\t\t%.6f (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6f (s)\n", systim);

    write_values(final_state_file, av_vels_file, params, cells, obstacles, av_vels);
    finalise(&cells, &tmp_cells, &obstacles, &av_vels);

    return EXIT_SUCCESS;
}

void write_values(const char * final_state_file, const char * av_vels_file,
    const param_t params, float* cells, int* obstacles, float* av_vels)
{
    FILE* fp;                     /* file pointer */
    int ii,jj,kk;                 /* generic counters */
    const float c_sq = 1.0/3.0;  /* sq. of speed of sound */
    float local_density;         /* per grid cell sum of densities */
    float pressure;              /* fluid pressure in grid cell */
    float u_x;                   /* x-component of velocity in grid cell */
    float u_y;                   /* y-component of velocity in grid cell */
    float u;                     /* norm--root of summed squares--of u_x and u_y */

    fp = fopen(final_state_file, "w");

    if (fp == NULL)
    {
        DIE("could not open file output file");
    }

    for (ii = 0; ii < params.ny; ii++)
    {
        for (jj = 0; jj < params.nx; jj++)
        {
            /* an occupied cell */
            if (obstacles[ii*params.nx + jj])
            {
                u_x = u_y = u = 0.0;
                pressure = params.density * c_sq;
            }
            /* no obstacle */
            else
            {
				int current_index = (ii*params.nx + jj)*9;
                local_density = 0.0;

                for (kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += cells[current_index + kk];
                }
				

                /* x-component of velocity */
                u_x = (cells[current_index + 1] +
                        cells[current_index + 5] +
                        cells[current_index + 8]
                    - (cells[current_index + 3] +
                        cells[current_index + 6] +
                        cells[current_index + 7])) /
                    local_density;

                /* compute y velocity component */
                u_y = (cells[current_index + 2] +
                        cells[current_index + 5] +
                        cells[current_index + 6]
                    - (cells[current_index + 4] +
                        cells[current_index + 7] +
                        cells[current_index + 8])) /
                    local_density;

                /* compute norm of velocity */
                u = sqrt((u_x * u_x) + (u_y * u_y));

                /* compute pressure */
                pressure = local_density * c_sq;
            }

            /* write to file */
            fprintf(fp,"%d %d %.12E %.12E %.12E %.12E %d\n",
                jj,ii,u_x,u_y,u,pressure,obstacles[ii*params.nx + jj]);
        }
    }

    fclose(fp);

    fp = fopen(av_vels_file, "w");
    if (fp == NULL)
    {
        DIE("could not open file output file");
    }

    for (ii = 0; ii < params.max_iters; ii++)
    {
        fprintf(fp,"%d:\t%.12E\n", ii, av_vels[ii]);
    }

    fclose(fp);
}

float calc_reynolds(const param_t params, float av)
{
    const float viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);

    return av * params.reynolds_dim / viscosity;
}

float total_density(const param_t params, float* cells)
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
