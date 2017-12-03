
#include "cuda_runtime.h"
#include "device_functions.h"
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <windows.h>
#include <queue>
#include <map>

#define CHECK_CUDA_STATUS(val) { printf("Err: %s: line %d, file %s\n", cudaGetErrorString((cudaError_t)val), __LINE__, __FILE__); } 

using namespace std;

inline unsigned __int64 GetCycle() {
	__asm _emit 0x0F
	__asm _emit 0x31
};

struct point_t {
	int			dimen;
	double		cor[2];
	int			idx;					// -1 noice, >= 0 cluster id
	int			vis;					// -1 unvisited, 1 core, 2 border

};

//__constant__ point_t pts[1000];

double __device__ cudaCalcDistance(const point_t &src, const point_t &dest) {
	double res = 0.0;
	for (int i = 0; i < src.dimen; i++) {
		res += (src.cor[i] - dest.cor[i]) * (src.cor[i] - dest.cor[i]);
	}
	return res;
}

/*			p0	p1	p2	p3	...	pn
 *	point0->	*	*	*	*	...	*
 *	point1->	*	*	*	*	...	*
 *	point2->	*	*	*	*	...	*
 *	point3->	*	*	*	*	...	* 
 *	  ...                       ...
 *	pointn->	*	*	*	*	...	*
 */
void __global__ cudaGetNeighbors(point_t* points, int len, int* neighbors, double minEps, int minPts) {
	
	unsigned int	tid	= blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int	src;
	unsigned int	dest;
	unsigned int	point_id = tid;
	unsigned int	neighborscnt;

	while (point_id < len * len) {
		src = point_id / len;
		dest = point_id % len;
		double dist = 0.0;
		if (src <= dest) {
			dist = cudaCalcDistance(points[src], points[dest]);
			if (dist < minEps * minEps) {
				neighbors[point_id] = 1;
			}
			neighbors[dest * len + src] = neighbors[point_id];
		}
		point_id += blockDim.x * gridDim.x;
	}

	__syncthreads();

	point_id = tid;
	while (point_id < len) {
		neighborscnt = 0;
		src = point_id * len;
		for (int i = 0; i < len; i++) {
			if (point_id != i) {
				if (neighbors[src + i]) {
					neighborscnt++;
				}
			}
		}
		if (neighborscnt >= minPts) {
			points[point_id].vis = 1;
		}
		point_id += blockDim.x * gridDim.x;
	}
}

void hostSetIdx(point_t* points, int len, int* hostNeighbors) {
	queue<int> s;
	int t_idx = 1;
	for (int i = 0; i < len; i++) {
		if (points[i].vis >= 0) {
			if (points[i].idx < 1) {
				points[i].idx = t_idx;
				int src = i * len;
				for (int j = 0; j < len; j++) {
					if (hostNeighbors[src + j]) {
						points[j].idx = t_idx;
						s.push(j);
					}
				}
				while (!s.empty()) {
					if (points[s.front()].vis == 1) {
						src = s.front() * len;
						for (int j = 0; j < len; j++) {
							if (hostNeighbors[src + j]) {
								if (points[j].idx < 1) {
									points[j].idx = t_idx;
									s.push(j);
								}
							}
						}
					}
					s.pop();
				}
			}
			//for (int i = 0; i < len; i++) cout << points[i].idx << " ";
			//cout << endl;
			t_idx++;
		}
	}
}

point_t* DBSCAN(point_t* points, int len, double minEps, int minPts) {
	int threadsNum;
	printf("CUDA threads number: ");
	cin >> threadsNum;

	int *hostNeighborArray = (int*)malloc(len * len * sizeof(int));
	for (int i = 0; i < len * len; i++) hostNeighborArray[i] = -1;

	point_t* cudaPoints;
	CHECK_CUDA_STATUS(cudaMalloc((void**)&cudaPoints, len * sizeof(point_t)));

	int *cudaNeighborArray;
	CHECK_CUDA_STATUS(cudaMalloc((void**)&cudaNeighborArray, len * len * sizeof(int)));

	CHECK_CUDA_STATUS(cudaMemcpy(cudaPoints, points, len * sizeof(point_t), cudaMemcpyHostToDevice));

	dim3 threads;
	threads.x = threadsNum;
	dim3 blocks;
	blocks.x = (len * len + threads.x - 1) / threads.x;

	cudaGetNeighbors << <blocks, threads >> > (cudaPoints, len, cudaNeighborArray, minEps, minPts);

	CHECK_CUDA_STATUS(cudaMemcpy(hostNeighborArray, cudaNeighborArray, len * len * sizeof(int), cudaMemcpyDeviceToHost));
	CHECK_CUDA_STATUS(cudaMemcpy(points, cudaPoints, len * sizeof(point_t), cudaMemcpyDeviceToHost));
	hostSetIdx(points, len, hostNeighborArray);
	/*
	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len; j++)
		{
			cout << hostNeighborArray[i * len + j] << " ";
		}
		cout << endl;
	}
	*/
	cudaFree(cudaPoints);
	cudaFree(cudaNeighborArray);
	return points;
}

int main(int argc, char* argv[]) {
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	ifstream fin("cluster.txt");
	//freopen("cluster.txt", "r", stdin);
	srand(time(0));
	point_t pts[312];
	int len = 312;
	for (int i = 0; i < len; i++) {
		fin >> pts[i].cor[0] >> pts[i].cor[1];
	}
	for (int i = 0; i < len; i++) {
		pts[i].dimen = 2;
		pts[i].vis = -1;
		pts[i].idx = -1;
	}
	for (int i = 0; i < len; i++) cout << pts[i].cor[0] << " " << pts[i].cor[1] << " " << pts[i].idx << endl;
	clock_t st;
	cudaEventRecord(start, 0);
	st = clock();
	DBSCAN(pts, len, 2.0, 3);
	printf("Parallel time: %d ms\n", clock() - st);
	cudaEventRecord(end, 0);
	float epstime;
	cudaEventElapsedTime(&epstime, start, end);
	//printf("Parallel time: %f ms\n", epstime);
	
	map <int, int> mp;
	for (int i = 0; i < len; i++) {
		cout << pts[i].cor[0] << " " << pts[i].cor[1] << " " << pts[i].idx << endl;
		mp[pts[i].idx]++;
	}
	
	map <int, int>::iterator it = mp.begin();
	for (; it != mp.end(); it++) cout << it->first << " " << it->second << endl;
	
	return 0;
}
