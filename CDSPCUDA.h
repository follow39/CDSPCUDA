/*
 * CDSPCUDA.h
 *
 *  Created on: 01 янв. 2020 г.
 *      Author: user
 */

#ifndef CDSPCUDA_H_
#define CDSPCUDA_H_

#include "CLog.h"
#include "CDSP.h"
#include "fftw3.h"
#ifdef QUASAR_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
//#include <cuda_runtime_api.h>
//#include <driver_types.h>
//#define __CUDA_RUNTIME_H__
//#define	__DRIVER_TYPES_H__
//#include <CUDA_DRIVER_API.h>
#include "./cuda/helper_cuda.h"
#endif
//#include <omp.h>

#define MAX_BUF_SIZE 	512

class CDSP_CUDA : public CDSP {
public:
	CDSP_CUDA(CLog *log);
	CDSP_CUDA();
	virtual ~CDSP_CUDA();
#ifdef QUASAR_CUDA
	void 	printDSPInfo(void);		// Вывод информации о параметрах системы
	//int fft_r2c(float* inData, fftwf_complex* outData, int N, int M);
	int removeDC(Ipp32fc* pImageData, int W0, int W, int H);
	int fft(Ipp32f* pImageData, int N, int M);
	int fft(Ipp32fc* pImageData, int N, int M);
	int ifft(Ipp32fc* pImageData, int N, int M);
	int fftshift(Ipp32fc* pImageData, int N, int M);
	int rmc(Ipp32fc* pData, int w, int h, int hb, float lambda, float tp, float v);
	int hamming(float* pImageData, int W, int H, int W0, int H0);
	int hamming(Ipp32fc* pImageData, int W, int H, int W0, int H0);
	int complexMul(Ipp32fc* SrcDst1, Ipp32fc* Src2, int len) const;
    int complexMul(Ipp32fc* SrcDst1, float* Src2, int len) const;
	int	complexSum(Ipp32fc* Src, Ipp32fc* Dst, int len) const;
	float entropy(Ipp32fc* pData, int W, int H);
	float entropyf(Ipp32f* pData, int W, int H);
	int transpose(Ipp32fc* pImageData, int N, int M);
	int phocus(Ipp32fc* pData, int W, int H, int Hmax, float Rb, float Tp, float lambda, float V, float Rmax, int Np);
	int pointTargetSignal(void* data, int Ns, int Fs, int Np, float H, float V, float X, float Y, float A, float F0, float DF, float Tp, float Mu, bool complex, bool symmetric_ramp);
	
	int HammingWindow(float* pImageData, int W0);
    int BlackmanWindow(float* pImageData, int W0);
    int NuttallWindow(float* pImageData, int W0);
    
	int windowWeighting(float* pImageData, float *window, int W, int W0, int H0);
	int windowWeighting(Ipp32fc* pImageData, float *window, int W, int W0, int H0);
    
	int hamming(float* pImageData, float *window, int W, int W0, int H0);
	int hamming(Ipp32fc* pImageData, float *window, int W, int W0, int H0);
    
    int bp(CImage* image, CRadar* radar, float* outdata, float* wnd, float* V, float x0, float y0, float dx, float dy, float* h, int nx, int ny, bool bCorr);
    int bp_strip(CImage* in, CImage* out, CRadar *radar, float *window, float *V, float x0, float y0, float dx, float *h, int nx, int line, bool bCorr);

private:
	int 	square(Ipp32f* pInData, Ipp32f* pOutData, int N, int M);
	int 	log10(Ipp32f* pInData, Ipp32f* pOutData, int N, int M);

	int 		devID;
	CUdevice 	cuDevice;
    
    

    
    
    //float bp_pixel(CImage* image, CRadar *radar, float *wnd, float* V, float x, float y, float* h, bool bCorr);

private:
	void 	printCPUFeatures(void);										// Вывод информации о параметрах процессора
#endif
};

#endif /* CDSPCUDA_H_ */
