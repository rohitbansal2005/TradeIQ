#include <iostream>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {

    // Ultra-fast Exponential Moving Average (EMA) O(N) using sliding window pointer arithmetic
    EXPORT void calculate_ema_cpp(double* prices, double* ema_out, int length, int window) {
        if (length <= 0) return;
        
        double multiplier = 2.0 / (window + 1.0);
        
        // Initialize the first EMA value to the first price
        ema_out[0] = prices[0];
        
        for (int i = 1; i < length; i++) {
            ema_out[i] = ((prices[i] - ema_out[i - 1]) * multiplier) + ema_out[i - 1];
        }
    }

}
