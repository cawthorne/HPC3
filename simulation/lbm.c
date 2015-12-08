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

#include "mpi.h"

#include "lbm.h"

/*
** main program:
** initialise, timestep loop, finalise
*/

/*
** main program:
** initialise, timestep loop, finalise
*/

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
            (cells[ii*params.nx + jj + params.nx*params.ny*4] - w1) > 0.0 &&
            (cells[ii*params.nx + jj + params.nx*params.ny*7] - w2) > 0.0 &&
            (cells[ii*params.nx + jj + params.nx*params.ny*8] - w2) > 0.0 )
            {
                /* increase 'north-side' densities */
                cells[ii*params.nx + jj + params.nx*params.ny*2] += w1;
                cells[ii*params.nx + jj + params.nx*params.ny*5] += w2;
                cells[ii*params.nx + jj + params.nx*params.ny*6] += w2;
                /* decrease 'south-side' densities */
                cells[ii*params.nx + jj + params.nx*params.ny*4] -= w1;
                cells[ii*params.nx + jj + params.nx*params.ny*7] -= w2;
                cells[ii*params.nx + jj + params.nx*params.ny*8] -= w2;
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
            (cells[ii*params.nx + jj + params.nx*params.ny*3] - w1) > 0.0 &&
            (cells[ii*params.nx + jj + params.nx*params.ny*6] - w2) > 0.0 &&
            (cells[ii*params.nx + jj + params.nx*params.ny*7] - w2) > 0.0 )
            {
                /* increase 'east-side' densities */
                cells[ii*params.nx + jj + params.nx*params.ny*1] += w1;
                cells[ii*params.nx + jj + params.nx*params.ny*5] += w2;
                cells[ii*params.nx + jj + params.nx*params.ny*8] += w2;
                /* decrease 'west-side' densities */
                cells[ii*params.nx + jj + params.nx*params.ny*3] -= w1;
                cells[ii*params.nx + jj + params.nx*params.ny*6] -= w2;
                cells[ii*params.nx + jj + params.nx*params.ny*7] -= w2;
            }
        }
    }
}


int main(int argc, char* argv[])
{
	
	  int rank;               /* 'rank' of process among it's cohort */ 
  int size;               /* size of cohort, i.e. num processes started */
  int flag;               /* for checking whether MPI_Init() has been called */
  int strlen;             /* length of a character array */
  enum bool {FALSE,TRUE}; /* enumerated type: false = 0, true = 1 */  
  char hostname[MPI_MAX_PROCESSOR_NAME];  /* character array to hold hostname running process */

  /* initialise our MPI environment */
  MPI_Init( &argc, &argv );

  /* check whether the initialisation was successful */
  MPI_Initialized(&flag);
  if ( flag != TRUE ) {
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }

  /* determine the hostname */
  MPI_Get_processor_name(hostname,&strlen);

  /* 
  ** determine the SIZE of the group of processes associated with
  ** the 'communicator'.  MPI_COMM_WORLD is the default communicator
  ** consisting of all the processes in the launched MPI 'job'
  */
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  
  /* determine the RANK of the current process [0:SIZE-1] */
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );


  /* finialise the MPI enviroment */
  MPI_Finalize();

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

    parse_args(argc, argv, &final_state_file, &av_vels_file, &param_file);

    initialise(param_file, &accel_area, &params, &cells, &tmp_cells, &obstacles, &av_vels);
	
	int ob_num = 0;
	int i;
	for (i=0;i<params.nx*params.ny;i++){
		ob_num += !obstacles[i];
	}
	
	
    /* iterate for max_iters timesteps */
    gettimeofday(&timstr,NULL);
    tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);
	int iiM;
	int ii,jj;

    const float w0C = 4.0/9.0;    /* weighting factor */
    const float w1C = 1.0/9.0;    /* weighting factor */
    const float w2C = 1.0/36.0;   /* weighting factor */
	
		    //float w1,w2;  /* weighting factors */

    /* compute weighting factors */
   // w1 = params.density * params.accel * w1C;
    //w2 = params.density * params.accel * w2C;

    for (iiM = 0; iiM < params.max_iters; iiM++)
    {
		accelerate_flow(params,accel_area,cells,obstacles);
		
		float avs = 0;
		
		
for (ii = 0;ii<params.ny;ii++){
	for (jj = 0;jj<params.nx;jj++){
			int x_e,x_w,y_n,y_s;
			/*
for (ii = (myrank)*params.ny/4;ii<(myrank+1)*params.ny/4;ii++){
	for (jj = 0;jj<params.nx;jj++){

	int x_e,x_w,y_n,y_s;
	
	
  float sendBottom[params.nx*9];
  float sendTop[params.nx*9];
  float recTop[params.nx*9];
  float recBottom[params.nx*9];
  
  for (int sideNum = 0;sideNum<params.nx;sideNum++){
	  for (int kk = 0;kk<9;kk++){
		  cells[0*params.nx + sideNum + params.nx*params.ny*kk];
	  }
	  
  }
  
    if ((myrank % 2) == 0) {

    MPI_Ssend(sendTop,sizeof(float)*params.nx*9, MPI_FLOAT, (myrank-1)%4, tag, MPI_COMM_WORLD);
	MPI_Ssend(sendBottom,sizeof(float)*params.nx*9, MPI_FLOAT, (myrank+1)%4, tag, MPI_COMM_WORLD);
	MPI_Recv(recBottom, sizeof(float)*params.nx*9, MPI_FLOAT, (myrank+1)%4, tag, MPI_COMM_WORLD, &status);
	MPI_Recv(recTop, sizeof(float)*params.nx*9, MPI_FLOAT, (myrank-1)%4, tag, MPI_COMM_WORLD, &status);

  }
  else {
   
	
    MPI_Recv(recBottom, sizeof(float)*params.nx*9, MPI_FLOAT, (myrank+1)%4, tag, MPI_COMM_WORLD, &status);
	MPI_Recv(recTop, sizeof(float)*params.nx*9, MPI_FLOAT, (myrank-1)%4, tag, MPI_COMM_WORLD, &status);
	MPI_Ssend(sendTop,sizeof(float)*params.nx*9, MPI_FLOAT, (myrank-1)%4, tag, MPI_COMM_WORLD);
	MPI_Ssend(sendBottom,sizeof(float)*params.nx*9, MPI_FLOAT, (myrank+1)%4, tag, MPI_COMM_WORLD);
  }
	*/
  

    float u_x,u_y;               /* av. velocities in x and y directions */
    float u_sq;                  /* squtmp1red velocity */
    float local_density;         /* sum of densities in a particular cell */

			y_n = (ii + 1 == params.ny) ? (0) : (ii + 1);
            x_e = (jj + 1 == params.nx) ? (0) : (jj + 1);
            y_s = (ii == 0) ? (params.ny - 1) : (ii - 1);
            x_w = (jj == 0) ? (params.nx - 1) : (jj - 1);
            /* propagate densities to neighbouring cells, following
            ** appropriate directions of travel and writing into
            ** scratch space grid */
			
			 float local_cell[9];
				
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
					local_cell[3]+ local_cell[4]+local_cell[5]+ 
				local_cell[6]+local_cell[7]+local_cell[8];
				
                /* compute x velocity component */
                u_x = 		(local_cell[1] + local_cell[5] + local_cell[8] - (local_cell[3] + 
				local_cell[6] + local_cell[7]))/local_density;

                /* compute y velocity component */
                u_y = 
				(local_cell[2] + local_cell[5] + local_cell[6]- (local_cell[4] + 
				local_cell[7] 
				+ local_cell[8]))/ local_density;

                /* velocity squtmp1red */
                u_sq = u_x * u_x + u_y * u_y;
                
				const float utmp1 =   u_x + u_y; 

                const float utmp2 = - u_x + u_y;

				const int b = obstacles[ii*params.nx + jj];
				
		
				tmp_cells[current_index + params.nx*params.ny*0] = b ? local_cell[0] : (local_cell[0] + params.omega * ( local_density * w0C * (1.0 - u_sq *  (1.5)) - local_cell[0]));
                
				tmp_cells[current_index + params.nx*params.ny*1] = b ? local_cell[3] : (local_cell[1] + params.omega * ( w1C * local_density * (1.0 + u_x * 3.0
                    + (u_x * u_x)*4.5 - u_sq *(1.5)) - local_cell[1]));
					
				tmp_cells[current_index + params.nx*params.ny*2] = b ? local_cell[4]:(local_cell[2] + params.omega * (w1C * local_density * (1.0 + u_y * 3.0
                    + (u_y * u_y)*4.5
                    - u_sq *(1.5)) - local_cell[2]));
					
				tmp_cells[current_index + params.nx*params.ny*3] = b ? local_cell[1]:(local_cell[3] + params.omega * (w1C * local_density * (1.0 - u_x		* 3.0
                    + (u_x * u_x)*4.5- u_sq *(1.5)) - local_cell[3]));
					
				tmp_cells[current_index + params.nx*params.ny*4] = b ? local_cell[2]:(local_cell[4] + params.omega * ( w1C * local_density * (1.0 - u_y * 3.0
                    + (u_y * u_y)*4.5- u_sq *(1.5)) - local_cell[4]));
				tmp_cells[current_index + params.nx*params.ny*5] = b ? local_cell[7]:(local_cell[5] + params.omega * (w2C * local_density * (1.0 + utmp1 * 3.0
                    + (utmp1 * utmp1)*4.5 - u_sq *(1.5)) - local_cell[5]));
					
				tmp_cells[current_index + params.nx*params.ny*6] = b ? local_cell[8]:(local_cell[6] + params.omega * (w2C * local_density * (1.0 + utmp2 * 3.0
                    + (utmp2 * utmp2)*4.5 - u_sq *(1.5)) - local_cell[6]));
				tmp_cells[current_index + params.nx*params.ny*7] = b ? local_cell[5]:(local_cell[7] + params.omega * (w2C * local_density * (1.0 - utmp1 * 3.0
                    + (utmp1 * utmp1)*4.5 - u_sq *(1.5)) - local_cell[7]));
					
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
                float av = b ? 0 : sqrt(u_x*u_x + u_y*u_y);

		avs += av;

			}
		}
		
		av_vels[iiM] = avs / (float)ob_num;
		float* tmp = cells;
		cells = tmp_cells;
		tmp_cells = tmp;
        #ifdef DEBUG
		printf("\navs %.12E\n",avs);
        printf("==timestep: %d==\n", iiM);
        printf("Reynolds: %.12E ",calc_reynolds(params,av_vels[iiM]));
		printf("Density %.12E\n", total_density(params, cells));
        #endif
		
    }


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

	float* mover = malloc(sizeof(float)*params.nx*params.ny*9);
	
	for (i= 0;i<params.nx*params.ny;i++){
		mover[i*9] = cells[i];
		mover[i*9 + 1]= cells[i + 1*params.ny*params.nx];
		mover[i*9 + 2]= cells[i + 2*params.ny*params.nx];
		mover[i*9 + 3]= cells[i + 3*params.ny*params.nx];
		mover[i*9 + 4]= cells[i + 4*params.ny*params.nx];
		mover[i*9 + 5]= cells[i + 5*params.ny*params.nx];
		mover[i*9 + 6]= cells[i + 6*params.ny*params.nx];
		mover[i*9 + 7]= cells[i + 7*params.ny*params.nx];
		mover[i*9 + 8]= cells[i + 8*params.ny*params.nx];
	}
	
    write_values(final_state_file, av_vels_file, params, mover, obstacles, av_vels);
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
