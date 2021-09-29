#include "CDSPCUDA.h"
#include <omp.h>

CDSP_CUDA::CDSP_CUDA(CLog *log) /*: CDSP(log)*/{
	Log = log;
//	cudaDeviceProp deviceProp;
//	devID = gpuGetMaxGflopsDeviceId();
	#ifdef QUASAR_CUDA
	devID = gpuGetMaxGflopsDeviceId();
	checkCudaErrors(cuDeviceGet(&cuDevice, devID));
	#endif
}

CDSP_CUDA::CDSP_CUDA() {
	// TODO Auto-generated constructor stub
}

CDSP_CUDA::~CDSP_CUDA() {
	// TODO Auto-generated destructor stub
}

#ifdef QUASAR_CUDA

/*!
   \brief Вывод информации о версии библиотеки, частоте процессора, количестве ядер и возможностях процессора

	\return		 Функция не возвращает значения
*/



void CDSP_CUDA::printDSPInfo(void){
	

	Log->LogPrintf((char*)"============\n");
    
    int n;
    cudaGetDeviceCount(&n);
    Log->LogPrintf("Количество вычислителей  CUDA: %d\n", n);
    
    char name[100];
	cuDeviceGetName(name, 100, cuDevice);
	Log->LogPrintf("Вычислитель \t\t CUDA [%d]: %s\n", devID, name);
    
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    Log->LogPrintf("Частота вычислителя\t %u кГц\n", prop.memoryClockRate );
	Log->LogPrintf("Количество ядер \t %u\n", 0);
    
}


int CDSP_CUDA::removeDC(Ipp32fc* pImageData, int W0, int W, int H){
	
	Ipp32fc summ;
	
	
	for(int x = 0; x < W; x++){
		
		summ.im = 0;
		summ.re = 0;
		for(int y = 0; y < H; y++){
			
			summ.im += pImageData[W0*y+x].im;
			summ.re += pImageData[W0*y+x].re;
			
		}
		
		summ.im /= H;
		summ.re /= H;
		
		for(int y = 0; y < H; y++){
			
			pImageData[W0*y+x].im -= summ.im;
			pImageData[W0*y+x].re -= summ.re;
			
		}
		
	}
	
	return 0;
	
}

//============================
//fft()
//============================
int CDSP_CUDA::fft(Ipp32f* pImageData, int N, int M){
    
    /*
	int order = log2(N);
	assert(N == pow(2,order)); // N должно быть степенью двойки

	int 		  batch = M;
	float 		 *devInputData;
	cufftComplex *devOutputData;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&devInputData), N*batch*sizeof(float)));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&devOutputData), (N/2+1)*batch*sizeof(cufftComplex)));

	checkCudaErrors(cudaMemcpy(devInputData, pImageData, N*batch*sizeof(float), cudaMemcpyHostToDevice));

	cufftHandle handle;
	int rank = 1;                        // --- 1D FFTs
	int n[] = { N };                	 // --- Size of the Fourier transform
	int istride = 1, ostride = 1;        // --- Distance between two successive input/output elements
	int idist = N, odist = (N / 2 + 1);  // --- Distance between batches
	int inembed[] = { 0 };               // --- Input size with pitch (ignored for 1D transforms)
	int onembed[] = { 0 };               // --- Output size with pitch (ignored for 1D transforms)

	checkCudaErrors(cufftPlanMany(&handle, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch));
	checkCudaErrors(cufftExecR2C(handle, devInputData, devOutputData));

	#pragma omp parallel for
	for(int j=0; j<M; j++){
		float *p = &pImageData[j*N];
		cufftComplex *dp = &devOutputData[j*(N/2+1)];
		checkCudaErrors(cudaMemcpy(p, dp, (N/2)*sizeof(cufftComplex), cudaMemcpyDeviceToHost));
	}

	checkCudaErrors(cufftDestroy(handle));
	checkCudaErrors(cudaFree(devInputData));
	checkCudaErrors(cudaFree(devOutputData));
	*/
    
    int order = log2(N);
	assert(N == pow(2,order)); // N должно быть степенью двойки
	
    fftwf_plan FFTPlan;
    
	if(!FFTPlan){
		FFTPlan = fftwf_plan_dft_r2c_1d(N,  NULL, NULL, FFTW_ESTIMATE );
	}
	
	#pragma omp parallel for
	for(int j=0; j<M; j++){
		float *p = &pImageData[j*N];
		p[0] = 0.0f; p[1] = 0.0f;
		fftwf_execute_dft_r2c(FFTPlan, p, (fftwf_complex*)p);
		p[0] = 0; p[1] = 0; // Убираем постоянную составляющую
	}
	

	return -1;
}

int CDSP_CUDA::fft(Ipp32fc* pImageData, int N, int M){
	int order = log2(N);
	assert(N == pow(2,order)); // N должно быть степенью двойки

	int nbthreads = omp_get_max_threads();
    fftwf_init_threads();
    fftwf_plan_with_nthreads(nbthreads);
	omp_set_num_threads(nbthreads);

    fftwf_plan p;
    int rank = 1;
    int n[] = { (int)N };
	
#pragma omp critical
    p = fftwf_plan_many_dft(rank, n, M, (fftwf_complex*)pImageData, NULL, 1, N, (fftwf_complex*)pImageData, NULL, 1, N, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

#pragma omp critical
    fftwf_destroy_plan(p);
  
/*	//#pragma omp parallel for
	for(int j=0; j<M; j++){
		fftwf_plan p1;
		fftwf_complex *p = (fftwf_complex*)(&pImageData[j*N]);

		//#pragma omp critical (make_plan)
		p1 = fftwf_plan_dft_1d(N,  p, (fftwf_complex*)p, FFTW_FORWARD, FFTW_ESTIMATE );

		fftwf_execute(p1);
		fftwf_destroy_plan(p1);
		
		fftwf_cleanup_threads();
	}
*/
	return -1;
}

int CDSP_CUDA::ifft(Ipp32fc* pImageData, int N, int M){
	int order = log2(N);
	assert(N == pow(2,order)); // N должно быть степенью двойки

	int nbthreads = omp_get_max_threads();
    fftwf_init_threads();
    fftwf_plan_with_nthreads(nbthreads);
	omp_set_num_threads(nbthreads);
	
	fftwf_plan p;
    int rank = 1;
    int n[] = { (int)N };
	
	#pragma omp critical
    p = fftwf_plan_many_dft(rank, n, M, (fftwf_complex*)pImageData, NULL, 1, N, (fftwf_complex*)pImageData, NULL, 1, N, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

	#pragma omp critical
    fftwf_destroy_plan(p);
	
	//#pragma omp parallel for
//	for(int j=0; j<M; j++){
//		fftwf_plan p1;
//		fftwf_complex *p = (fftwf_complex*)(&pImageData[j*N]);

		//#pragma omp critical (make_plan)
//		p1 = fftwf_plan_dft_1d(N,  p, (fftwf_complex*)p, FFTW_BACKWARD, FFTW_ESTIMATE );

//		fftwf_execute(p1);
//		fftwf_destroy_plan(p1);
//	}

	return -1;
}

//============================
//fftshift()
//============================
int CDSP_CUDA::fftshift(Ipp32fc* pImageData, int N, int M){
	#pragma omp parallel for
	for(int j=0; j<M; j++){
		//fftshift
		Ipp32fc *pCopyBuffer= (Ipp32fc*) malloc( N/2 * sizeof(Ipp32fc));
		memcpy(pCopyBuffer, &pImageData[j*N], N/2*sizeof(Ipp32fc));
		memcpy(&pImageData[j*N], &pImageData[j*N + N/2], N/2*sizeof(Ipp32fc));
		memcpy(&pImageData[j*N + N/2], pCopyBuffer, N/2*sizeof(Ipp32fc));
		free(pCopyBuffer);
	}

	return 0;
}

//==================================
// rmc()
//==================================
int CDSP_CUDA::rmc(Ipp32fc* pData, int W, int H, int Hb, float Lambda, float Tp, float V){
	float 	Fdmax = 1/(2.0f*Tp);
	float	dFd = 2*Fdmax/W;
	float	a = Lambda/(2.0f*V);

	#pragma omp parallel for
	for (int w = 0; w < W; w++){
		float fd = -Fdmax + w*dFd;
		float k = sqrt(1.0 - pow(a*fd,2));

		if(k>1) k = 1;

		for(int h=0; h<H; h++){
			//int index = round(h/k);
			int index = round((h+Hb)/k - Hb);
			if(index >= H)
				continue;

			pData[h*W + w] = pData[index*W + w];
		}
	}

	return 0;
}



int CDSP_CUDA::HammingWindow(float* pImageData, int W0){

	#pragma omp parallel for
	for(int i=0; i<W0; i++){
		pImageData[i]=0.54f-0.46f*cos(2*M_PI*i/W0);
	}

	return 0;
}


int CDSP_CUDA::BlackmanWindow(float* pImageData, int W0){
    
    const float a0 = 0.42f;
    const float a1 = 0.5f;
    const float a2 = 0.08f;
    
	#pragma omp parallel for
	for(int i=0; i<W0; i++){
		pImageData[i]=a0-a1*cos(2*M_PI*i/W0)+a2*cos(4*M_PI*i/W0);
	}

	return 0;
}


int CDSP_CUDA::NuttallWindow(float* pImageData, int W0){
    
    const float a0 = 0.355768f;
    const float a1 = 0.487396f;
    const float a2 = 0.134232f;
    const float a3 = 0.012604f;
    
	#pragma omp parallel for
	for(int i=0; i<W0; i++){
		pImageData[i]=a0-a1*cos(2*M_PI*i/W0)+a2*cos(4*M_PI*i/W0)-a3*cos(6*M_PI*i/W0);
	}

	return 0;
}

int CDSP_CUDA::windowWeighting(float* pImageData, float *window, int W, int W0, int H0){

	#pragma omp parallel for
	for(int i = 0; i<H0; i++){
		for(int j = 0; j<W0; j++){
			pImageData[i*W + j]*=window[j];
		}
	}
	return 0;
}

int CDSP_CUDA::windowWeighting(Ipp32fc* pImageData, float *window, int W, int W0, int H0){
	#pragma omp parallel for
	for(int i = 0; i<H0; i++){
		for(int j = 0; j<W0; j++){
			pImageData[i*W + j].im *=window[j];
			pImageData[i*W + j].re *=window[j];
		}
	}
	return 0;
}

int CDSP_CUDA::hamming(float* pImageData, float *window, int W, int W0, int H0){

	#pragma omp parallel for
	for(int i = 0; i<H0; i++){
		for(int j = 0; j<W0; j++){
			pImageData[i*W + j]*=window[j];
		}
	}
	return 0;
}

int CDSP_CUDA::hamming(Ipp32fc* pImageData, float *window, int W, int W0, int H0){
	#pragma omp parallel for
	for(int i = 0; i<H0; i++){
		for(int j = 0; j<W0; j++){
			pImageData[i*W + j].im *=window[j];
			pImageData[i*W + j].re *=window[j];
		}
	}
	return 0;
}


int CDSP_CUDA::hamming(float* pImageData, int W, int H, int W0, int H0){
	float *window = (float*)malloc(W0*sizeof(float));

	#pragma omp parallel for
	for(int i=0; i<W0; i++){
		window[i]=0.54f-0.46f*cos(2*M_PI*i/W0);
	}

	#pragma omp parallel for
	for(int i = 0; i<H0; i++){
		for(int j = 0; j<W0; j++){
			pImageData[i*W + j]*=window[j];
		}
	}

	free(window);
	return 0;
}

int CDSP_CUDA::hamming(Ipp32fc* pImageData, int W, int H, int W0, int H0){
	float *window = (float*)malloc(W0*sizeof(float));

	#pragma omp parallel for
	for(int i=0; i<W0; i++){
		window[i]=0.54f-0.46f*cos(2*M_PI*i/W0);
	}

	#pragma omp parallel for
	for(int i = 0; i<H0; i++){
		for(int j = 0; j<W0; j++){
			pImageData[i*W + j].im *=window[j];
			pImageData[i*W + j].re *=window[j];
		}
	}

	free(window);
	return 0;
}

int CDSP_CUDA::complexMul(Ipp32fc* SrcDst1, Ipp32fc* Src2, int len) const{
	
	
	for(int i = 0; i<len; i++){
		float re = SrcDst1[i].re*Src2[i].re - SrcDst1[i].im*Src2[i].im;
		float im = SrcDst1[i].re*Src2[i].im + SrcDst1[i].im*Src2[i].re;
		SrcDst1[i].re = re;
		SrcDst1[i].im = im;
	}
	

	return 0;
}

int CDSP_CUDA::complexMul(Ipp32fc* SrcDst1, float* Src2, int len) const{
	for(int i = 0; i<len; i++){
		SrcDst1[i].re = SrcDst1[i].re*Src2[i];
		SrcDst1[i].im = SrcDst1[i].im*Src2[i];
	}

	return 0;
}

int	CDSP_CUDA::complexSum(Ipp32fc* Src, Ipp32fc* Dst, int len) const {
	
	Dst[0].re = 0;
	Dst[0].im = 0;
	for(int i = 0; i<len; i++){
		Dst[0].re += Src[i].re;
		Dst[0].im += Src[i].im;
	}

	return 0;
}

//==================================
// entropyf() Вычисляет энергию вектора
//==================================
float CDSP_CUDA::entropyf(float* pData, int W, int H){
    
    
    printf("ENT\n");
    
	float P = 0;
	//#pragma omp parallel for
	for(int i=0; i<W*H; i++){
		P += pData[i]*pData[i];
	}

	float E = 0;
	//#pragma omp parallel for
	for(int i=0; i<W*H; i++){
		float p = pData[i]*pData[i]/P;
		E += p*log(p);
	}

	return -E;
}

float CDSP_CUDA::entropy(Ipp32fc* pData, int W, int H){
    
    printf("ENT\n");
    
	float P = 0;
	//#pragma omp parallel for
	for(int i=0; i<W*H; i++){
		float m = sqrt(pData[i].re*pData[i].re + pData[i].im*pData[i].im);
		P += m*m;
	}

	float E = 0;
	//#pragma omp parallel for
	for(int i=0; i<W*H; i++){
		float m = sqrt(pData[i].re*pData[i].re + pData[i].im*pData[i].im);
		float p = m*m/P;
		E += p*log(p);
	}

	return -E;
}

//==================================
// transpose()
//==================================
int CDSP_CUDA::transpose(Ipp32fc* pImageData, int N, int M){
	int buflen = N*M*sizeof(Ipp32fc);
	Ipp32fc* buff = (Ipp32fc*)aligned_alloc(64, buflen);
	memset(buff, 0, buflen);

	// FIXME сделать блочное
	for(int j = 0; j<M; j++){
		for(int i = 0; i<N; i++){
			buff[M*i + j] = ((Ipp32fc*)pImageData)[N*j + i];
		}
	}

	memcpy(pImageData, buff, buflen);
	free(buff);

	return 0;
}

//==================================
// phocus()
//==================================
int CDSP_CUDA::phocus(Ipp32fc* pData, int W, int H, int Hmax, float Rb, float Tp, float lambda, float V, float Rmax, int Np){
	#pragma omp parallel for
	for (int h = 0; h < H; h++){
		float r = Rb + Rmax/Hmax*h;
		if(r == 0) r = Rb + Rmax/Hmax;
		float c1 = 2*M_PI/lambda/r;
		// Формирование опорной функции
		Ipp32fc* S = (Ipp32fc*)aligned_alloc(64, W*sizeof(Ipp32fc));
		memset(S, 0, W*sizeof(Ipp32fc)); // Это обязательно
		for (int w=0; w<Np; w++){
			float t = w - Np/2.0f;
			float phase = -c1*pow(V*t*Tp,2);
			S[w].im = sin(phase);
			S[w].re = cos(phase);
		}

		complexMul(&pData[h*W], S, W);
		free(S);
	}

	return 0;
}

int CDSP_CUDA::pointTargetSignal(void* data, int Ns, int Fs, int Np, float H, float V, float X, float Y, float A, float F0, float DF, float Tp, float Mu, bool complex, bool symmetric_ramp){
	float c1 = 1.0f/Fs;
	float c2 = pow(X, 2) + pow(H, 2);

	float f0 = F0;
	float mu = Mu;

	#pragma omp parallel for	// Почему-то сигнал рваный получается
	for(int i=0; i<Np; i++){
		if(symmetric_ramp && (i & 1)){
			f0 = F0 + DF;
			Mu = - Mu;
		} else {
			f0 = F0;
			mu = Mu;
		}

		float t, R, p1, p2, p3, phase;
		for(int j=0; j<Ns; j++){
			// Вычисление фазы опорной функции для каждого периода
			t = (float)j*c1;	// Отсчеты времени в пределах периода
			R = sqrt(pow((Y + V * (t + i*Tp)), 2) + c2);
			p1 = f0;
			p2 = mu*t;
			p3 = -mu*R/SPEED_OF_LIGHT;
			phase = 4*M_PI*R/SPEED_OF_LIGHT*(p1 + p2 + p3);
			if(!complex)
				((Ipp32f*)data)[i*Ns + j] += A*cos(phase) + 1;
			else{
				(((Ipp32fc*)data)[i*Ns + j]).re += A*cos(phase) + 1;
				(((Ipp32fc*)data)[i*Ns + j]).im += A*sin(phase) + 1;
			}
		}
	}
	return 0;
}




__device__ inline void complexMul_(Ipp32fc* Dst, const Ipp32fc Src){
	float re = 0;
	float im = 0;
	re = Dst[0].re*Src.re - Dst[0].im*Src.im;
	im = Dst[0].re*Src.im + Dst[0].im*Src.re;
	Dst[0].re = re;
	Dst[0].im = im;

}

__device__ inline void complexMul_(Ipp32fc* SrcDst1, const float Src2){

    SrcDst1[0].re = SrcDst1[0].re*Src2;
    SrcDst1[0].im = SrcDst1[0].im*Src2;

}

__device__ inline void complexSum_(Ipp32fc* Dst, const Ipp32fc Src){
	Dst[0].re += Src.re;
	Dst[0].im += Src.im;
}


__device__ void bpPixel(Ipp32fc* indata, float* outdata, unsigned int i, unsigned int j,
    
    float *V, float x0, float y0, float dx, float dy, float *h, int nx, int ny, bool bCorr,
    float dt,
	float c1,
	float c2,
	float c5,
	float dfr,
	float dr,
    int Np,
    int Ns,
    float *devWindow){
	

	float y = y0 + i * dy;
	float x = x0 + j * dx;
    
	float t = 0;
	float r = 0;
	int ind = 0;
	float phase = 0;

	float fp = 0;
	float fr = 0;
	float phi = 0;
    float c4 = 0;

	Ipp32fc Sum;
	Sum.re = 0;
	Sum.im = 0;

	Ipp32fc signal;
	Ipp32fc sop;
	Ipp32fc sk;


	for (int k = 0; k < Np; k++){
        c4 = h[k]*h[k] + x * x;
		t = k * dt;
		r = sqrtf(c4 + pow((y + V[k] * t), 2));
		ind = round(r / dr);

		signal = indata[k * Ns + ind];
		phase = c1 * r;

		sop.re = cosf(phase);
		sop.im = sinf(phase);

		if (bCorr)
		{
			fp = (ind - 1)*dfr;

			fr = r * c5;
			phi = c2 * (fr - fp);
			sk.re = cosf(phi);
			sk.im = sinf(phi);

			complexMul_(&signal, sk);
		}

        complexMul_(&signal, devWindow[k]); // Оконное взвешивание
		complexMul_(&signal, sop);
        complexSum_(&Sum, signal);

	}

	*outdata = sqrtf(Sum.im*Sum.im + Sum.re*Sum.re);
}


__global__ void bp_(Ipp32fc* indata, float* outdata,
    float *V, float x0, float y0, float dx, float dy,
    float *h, int nx, int ny, bool bCorr,
    float dt, float c1, float c2, float c5,
	float dfr, float dr, int Np, int Ns, float *devWindow
){
    int ind_i = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_y = blockDim.y * gridDim.y;
    
    int ind_j = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    
    float out = 0;
    
    for (int i = ind_i ; i < ny; i += stride_y){
        
        for (int j = ind_j ; j < nx; j += stride_x){
            bpPixel(indata, &out, i, j,
                    V, x0, y0, dx, dy, h, nx, ny, bCorr,
                    dt, c1, c2, c5, dfr, dr, Np, Ns, devWindow);
            outdata[i*nx + j] = out;
        }

    }

}



#define BLOCK_DIVISION 8

int CDSP_CUDA::bp(CImage* image, CRadar* radar, float* outdata, float* wnd, float* V, float x0, float y0, float dx, float dy, float* h, int nx, int ny, bool bCorr){
    
	float *window = (float*)malloc(image->ImageH0*sizeof(float));
	HammingWindow(window, image->ImageH0);

	int Np = image->ImageH0;
	int Ns = image->ImageWidth;
	float Tp = radar->GetTp();
	float Mu = radar->GetMu();
	float F0 = radar->F0;
	float Fs = radar->Fs;
    
    
    
    float dt = Tp;
	float c1 = -4*M_PI/SPEED_OF_LIGHT*F0;
	float c2 = -M_PI*Tp;
	float c5 = 2*Mu/SPEED_OF_LIGHT;
	float dfr = Fs/(2*(Ns-1));
	float dr = Fs * SPEED_OF_LIGHT / (4 * Mu * (Ns - 1));
    

	Ipp32fc* indata = (Ipp32fc*)image->pImageData;

	Ipp32fc* devInData;
	float*  devOutData;
	float* devWindow;
	float* devV;
	float* devH;
    
    cudaError_t cudaStatus;
    
    
    // In
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&devInData), Ns*Np*sizeof(Ipp32fc)));
	checkCudaErrors(cudaMemcpy(devInData, indata, Ns*Np*sizeof(Ipp32fc), cudaMemcpyHostToDevice));
    // Out
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&devOutData), nx*ny*sizeof(float)));
	checkCudaErrors(cudaMemcpy(devOutData, outdata, nx*ny*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(devOutData, 0, nx*ny*sizeof(float)));
    // Window
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&devWindow), image->ImageH0*sizeof(float)));
	checkCudaErrors(cudaMemcpy(devWindow, window, image->ImageH0*sizeof(float), cudaMemcpyHostToDevice));
	// V H
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&devV), Np*sizeof(float)));
	checkCudaErrors(cudaMemcpy(devV, V, Np*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&devH), Np*sizeof(float)));
	checkCudaErrors(cudaMemcpy(devH, h, Np*sizeof(float), cudaMemcpyHostToDevice));
    
    
    dim3 blocks((nx + BLOCK_DIVISION - 1)/BLOCK_DIVISION,
                      (ny + BLOCK_DIVISION - 1)/BLOCK_DIVISION);
	dim3 threads(BLOCK_DIVISION, BLOCK_DIVISION);
    
    
    bp_<<<blocks, threads>>>(devInData, devOutData, devV, x0, y0, dx, dy, devH, nx, ny, bCorr,
                            dt, c1, c2, c5, dfr, dr, Np, Ns, devWindow );

       
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching bpKernel!\n", cudaStatus);
	}

    checkCudaErrors(cudaMemcpy(outdata, devOutData, nx*ny*sizeof(float), cudaMemcpyDeviceToHost));
	
	checkCudaErrors(cudaFree(devInData));
	checkCudaErrors(cudaFree(devOutData));
	checkCudaErrors(cudaFree(devWindow));
	checkCudaErrors(cudaFree(devV));
	checkCudaErrors(cudaFree(devH));

	free(window);
    
	return 0;
}


int CDSP_CUDA::bp_strip(CImage* in, CImage* out, CRadar* radar, float *window, float *V, float x0, float y0, float dx, float *h, int nx, int line, bool bCorr){
	
	return 0;
}


#endif
