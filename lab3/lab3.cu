#include "lab3.h"
#include <cstdio>

// __device__: called from GPU, for GPU
// __device__ __host__: both device and host visible
__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }


// __global__: called from CPU, for GPU
__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) { // inside the mask
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void PoissonImageEditing_Jacobi(
	float *nowX, float *nextX,
	const float *background,
	const float *target,
	const float *mask,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;

	// locate pixel
	const int yb = oy+yt, xb = ox+xt;
	const int curb = wb*yb+xb;

	// determind the neightbor condition, true if interior, false if boundary
	// not considering cases that target is at the background edge
	const int Nb = curb - wb;
	const int Sb = curb + wb;
	const int Wb = curb - 1;
	const int Eb = curb + 1;

	const int Nt = curt - wt;
	const int St = curt + wt;
	const int Wt = curt - 1;
	const int Et = curt + 1;

	//bool IamBoundary_of_target = false;
	//bool IamBoundary_of_mask = true;
	bool IamInterior_of_mask = false;

	if ((xt == 0 or xt >= wt-1 or yt == 0 or yt >= ht-1)) { // at the boundary of target
		//IamBoundary_of_target = true;
		//if (mask[curt] > 127.0f) IamBoundary_of_mask = true;
	} else { // at the interior of target
		// the four surrounding pixel are guarranteed to be valid
		if ((mask[curt] > 127.0f) and (mask[Nt] > 127.0f) and (mask[St] > 127.0f) and (mask[Et] > 127.0f) and (mask[St] > 127.0f)){
			IamInterior_of_mask = true;
		}
	}

	if (!IamInterior_of_mask) { // nextX should have the same value as background
		nextX[curb*3+0] = background[curb*3+0];
		nextX[curb*3+1] = background[curb*3+1];
		nextX[curb*3+2] = background[curb*3+2];
	} else { // need to solve the linear system

		// check boundary constraint of 4 neighbors
		bool N_is_boundary = true;
		bool S_is_boundary = true;
		bool W_is_boundary = true;
		bool E_is_boundary = true;

		if (xt > 0 and xt < wt-1 and yt-1 > 0 and yt-1 < ht-1) { // not at boundary of target // north: y-1
			if ((mask[Nt-1] > 127.0f) and (mask[Nt+1] > 127.0f) and (mask[Nt-wt] > 127.0f) and (mask[Nt+wt] > 127.0f))
				N_is_boundary = false;
		}
		if (xt > 0 and xt < wt-1 and yt+1 > 0 and yt+1 < ht-1) { // not at boundary of target // south: y+1
			if ((mask[St-1] > 127.0f) and (mask[St+1] > 127.0f) and (mask[St-wt] > 127.0f) and (mask[St+wt] > 127.0f))
				S_is_boundary = false;
		}
		if (xt-1 > 0 and xt-1 < wt-1 and yt > 0 and yt < ht-1) { // not at boundary of target // west: x-1
			if ((mask[Wt-1] > 127.0f) and (mask[Wt+1] > 127.0f) and (mask[Wt-wt] > 127.0f) and (mask[Wt+wt] > 127.0f))
				W_is_boundary = false;
		}
		if (xt+1 > 0 and xt+1 < wt-1 and yt > 0 and yt < ht-1) { // not at boundary of target // east: x+1
			if ((mask[Et-1] > 127.0f) and (mask[Et+1] > 127.0f) and (mask[Et-wt] > 127.0f) and (mask[Et+wt] > 127.0f))
				E_is_boundary = false;
		}
		
		/***
		solve the linear system Ax = b, with Jacobi iteration
		***/

		for (int i=0; i<3; i++) { // for RGB channels
			//nextX[curb*3+i] = 4*background[curb*3+i] - background[Nb*3+i] - background[Sb*3+i] - background[Wb*3+i] - background[Eb*3+i];
            nextX[curb*3+i] = 4*target[curt*3+i] - target[Nt*3+i] - target[St*3+i] - target[Wt*3+i] - target[Et*3+i];
			if (N_is_boundary) nextX[curb*3+i] += background[Nb*3+i];
			if (S_is_boundary) nextX[curb*3+i] += background[Sb*3+i];
			if (W_is_boundary) nextX[curb*3+i] += background[Wb*3+i];
			if (E_is_boundary) nextX[curb*3+i] += background[Eb*3+i];

			/*if (!N_is_boundary) nextX[curb*3+i] += nowX[((oy+Nt/wt)*wb+ox+Nt%wt)*3+i];
			if (!S_is_boundary) nextX[curb*3+i] += nowX[((oy+St/wt)*wb+ox+St%wt)*3+i];
			if (!W_is_boundary) nextX[curb*3+i] += nowX[((oy+Wt/wt)*wb+ox+Wt%wt)*3+i];
			if (!E_is_boundary) nextX[curb*3+i] += nowX[((oy+Et/wt)*wb+ox+Et%wt)*3+i];*/
			
            if (!N_is_boundary) nextX[curb*3+i] += nowX[Nb*3+i];
			if (!S_is_boundary) nextX[curb*3+i] += nowX[Sb*3+i];
			if (!W_is_boundary) nextX[curb*3+i] += nowX[Wb*3+i];
			if (!E_is_boundary) nextX[curb*3+i] += nowX[Eb*3+i];

			nextX[curb*3+i] = nextX[curb*3+i] / 4.0f;
		}
	}
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox,
	const int max_iter
)
{
	//float *X = (float *)malloc(wb*hb*sizeof(float)*3);
    float *X_d;
    cudaMalloc(&X_d, wb*hb*sizeof(float)*3);
	/*for (int i=0; i<wb*hb*3; i++) {
		X[i] = 0.0f;
	}*/
	cudaMemcpy(X_d, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice); // initialize output by background
	int iter = 0;
	for (iter=0; iter<max_iter; iter++) {
		if (iter%2) { // alter between now and next
			PoissonImageEditing_Jacobi<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
				X_d, output,
				background, target, mask,
				wb, hb, wt, ht, oy, ox
			);
		} else {
			PoissonImageEditing_Jacobi<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
				output, X_d,
				background, target, mask,
				wb, hb, wt, ht, oy, ox
			);
		}
	}
	if (!(iter%2)) {
		cudaMemcpy(output, X_d, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    }
}
