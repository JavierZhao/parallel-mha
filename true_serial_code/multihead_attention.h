#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include <vector>
#include <cmath>
#include "matrix.h"


class MultiHeadAttention {
public:
    MultiHeadAttention(int num_heads, int d_model);

    Matrix compute(const Matrix& x);

    void set_weights(const Matrix &Wq_in, const Matrix &Wk_in, const Matrix &Wv_in, const Matrix &Wo_in);
    void set_biases(const Matrix &bq_in, const Matrix &bk_in, const Matrix &bv_in, const Matrix &bo_in);

private:
    int num_heads;
    int d_model;
    int depth;

    std::vector<Matrix> Wq;
    std::vector<Matrix> Wk;
    std::vector<Matrix> Wv;
    Matrix Wo;

    std::vector<Matrix> bq;
    std::vector<Matrix> bk;
    std::vector<Matrix> bv;
    Matrix bo;

    std::vector<Matrix> split_heads(const Matrix& x);
    Matrix scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V);
    Matrix softmax(const Matrix& x);
};

#endif // MULTIHEAD_ATTENTION_H
