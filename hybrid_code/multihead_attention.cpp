#include "multihead_attention.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <omp.h>

// Constructor
MultiHeadAttention::MultiHeadAttention(int num_heads, int d_model)
    : num_heads(num_heads), d_model(d_model), depth(d_model / num_heads),
      Wq(num_heads, Matrix(d_model, depth/d_model)), // Assuming depth is correctly calculated
      Wk(num_heads, Matrix(d_model, depth/d_model)),
      Wv(num_heads, Matrix(d_model, depth/d_model)),
      Wo(d_model, d_model),
      bo(1, depth) { // Update according to actual required dimensions
    Wq.resize(num_heads);
    Wk.resize(num_heads);
    Wv.resize(num_heads);
    bq.resize(num_heads);
    bk.resize(num_heads);
    bv.resize(num_heads);


    for (int i = 0; i < num_heads; ++i) {
        Wq[i] = Matrix::random(d_model, depth);
        Wk[i] = Matrix::random(d_model, depth);
        Wv[i] = Matrix::random(d_model, depth);
        bq[i] = Matrix::random(1, depth); 
        bk[i] = Matrix::random(1, depth);
        bv[i] = Matrix::random(1, depth);
    }

    Wo = Matrix::random(d_model, d_model);
    bo = Matrix::random(1, depth);
}



void MultiHeadAttention::set_weights(const Matrix& Wq_in, const Matrix& Wk_in, const Matrix& Wv_in, const Matrix& Wo_in) {
    for (int i = 0; i < num_heads; ++i) {
        Wq[i] = Wq_in.transpose().block(0, i * depth, d_model, depth);
        Wk[i] = Wk_in.transpose().block(0, i * depth, d_model, depth);
        Wv[i] = Wv_in.transpose().block(0, i * depth, d_model, depth);
    }  
    Wo = Wo_in.transpose();
}

void MultiHeadAttention::set_biases(const Matrix& bq_in, const Matrix& bk_in, const Matrix& bv_in, const Matrix& bo_in) {
    for (int i = 0; i < num_heads; ++i) {
        bq[i] = bq_in.block(0, i * depth, 1, depth);
        bk[i] = bk_in.block(0, i * depth, 1, depth);
        bv[i] = bv_in.block(0, i * depth, 1, depth);
    }
    bo = bo_in;
}

std::vector<Matrix> MultiHeadAttention::split_heads(const Matrix& x) {
    std::vector<Matrix> heads(num_heads, Matrix(x.rows, depth));
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < num_heads; ++i) {
        for (int row = 0; row < x.rows; ++row) {
            for (int col = 0; col < depth; ++col) {
                heads[i].data[row][col] = x.data[row][i * depth + col];
            }
        }
    }
    return heads;
}

Matrix MultiHeadAttention::scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V) {
    Matrix kt = K.transpose();
    Matrix qk = Matrix::multiply(Q, kt);

    double dk = static_cast<double>(K.cols);
    #pragma omp parallel for
    for (auto& row : qk.data)
        for (auto& val : row)
            val /= std::sqrt(dk);

    Matrix attention_weights = softmax(qk);
    return Matrix::multiply(attention_weights, V);
}

Matrix MultiHeadAttention::softmax(const Matrix& x) {
    Matrix result(x.rows, x.cols);
    // openMP
    #pragma omp parallel for 
    for (int i = 0; i < x.rows; ++i) {
        double max_val = *std::max_element(x.data[i].begin(), x.data[i].end());
        double sum = 0.0;

        for (int j = 0; j < x.cols; ++j) {
            result.data[i][j] = std::exp(x.data[i][j] - max_val);
            sum += result.data[i][j];
        }

        for (int j = 0; j < x.cols; ++j) {
            result.data[i][j] /= sum;
        }
    }
    return result;
}

Matrix MultiHeadAttention::compute(const Matrix& x) {
    std::vector<Matrix> heads = split_heads(x);
    std::vector<Matrix> Q_heads(num_heads), K_heads(num_heads), V_heads(num_heads);

    // std::cout << "Obtaining QKV: \n";
    #pragma omp parallel for
    for (int i = 0; i < num_heads; ++i) {
        Q_heads[i] = Matrix::multiply(x, Wq[i]);
        // std::cout << "finished Q\n";
        K_heads[i] = Matrix::multiply(x, Wk[i]);
        V_heads[i] = Matrix::multiply(x, Wv[i]);
        // Apply broadcast and add biases
        // std::cout << "In obtaining QKV, Adding Matrices with dimensions: " 
        //   << Q_heads[i].rows << "x" << Q_heads[i].cols << " and " 
        //   << bq[i].broadcast(Q_heads[i].rows).rows << "x" << bq[i].broadcast(Q_heads[i].rows).cols << std::endl;
        Q_heads[i] += bq[i].broadcast(Q_heads[i].rows);
        K_heads[i] += bk[i].broadcast(K_heads[i].rows);
        V_heads[i] += bv[i].broadcast(V_heads[i].rows);
    }

    // std::cout << "Checking Q: \n";
    // for (int i = 0; i < num_heads; ++i) {
	// std::cout << Q_heads[i] << std::endl;
    // }

    // std::cout << "Checking K: \n";
    // for (int i = 0; i < num_heads; ++i) {
	// std::cout << K_heads[i] << std::endl;
    // }

    // std::cout << "Checking V: \n";
    // for (int i = 0; i < num_heads; ++i) {
	// std::cout << V_heads[i] << std::endl;
    // }

    std::vector<Matrix> attention_heads(num_heads);
    // std::cout << "Calculating attention_heads: \n";
    #pragma omp parallel for
    for (int i = 0; i < num_heads; ++i) {
        attention_heads[i] = scaled_dot_product_attention(Q_heads[i], K_heads[i], V_heads[i]);
    }

    // std::cout << "Checking attention_heads: \n";
    // for (int i = 0; i < num_heads; ++i) {
    //     std::cout << attention_heads[i] << std::endl;
    // }        

    Matrix concat_attention(x.rows, num_heads * depth);
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < num_heads; ++i) {
        for (int row = 0; row < x.rows; ++row) {
            for (int col = 0; col < depth; ++col) {
                concat_attention.data[row][i * depth + col] = attention_heads[i].data[row][col];
            }
        }
    }

    // std::cout << "Calculating output: \n";
    Matrix output = Matrix::multiply(concat_attention, Wo);
    // print dimensions of output and bo
    // std::cout << "Calculating output. Adding Matrices with dimensions: " 
    //       << output.rows << "x" << output.cols << " and " 
    //       << bo.rows << "x" << bo.cols << std::endl;
    output += bo.broadcast(output.rows);
    return output;
}