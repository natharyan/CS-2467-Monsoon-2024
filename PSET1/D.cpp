/* 
Dilated Convolution in Pytorch for CPU memory

- Dilated Convolution is the same as the traditional convolution, except it uses gaps filled between kernel elements.
- 1-dilated convolution will be the same traditional convolution, 2-dilated convolution will one unit gap between each kernel element, and so on.
- Adding gaps increases the receptive field of the filter, so it can extract both local and global features.
- This reduces overfitting while maintaining the kernel size and the spacial resolution of the output image.
*/

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/DilatedConvolutionUtils.h>
#include <ATen/native/im2col.h>
#include <ATen/native/vol2col.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <tuple>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/slow_conv_dilated2d_native.h>
#include <ATen/ops/slow_conv_dilated3d_native.h>
#endif

namespace at::native {
namespace {

// convert the n-dimensional volume to columns
template <typename Dtype, int64_t dim>
void hvol2col(
    const Dtype* data_hvol,
    const int channels,
    const IntArrayRef input_size,
    const IntArrayRef output_size,
    const IntArrayRef kernel_size,
    const IntArrayRef stride_size,
    const IntArrayRef pad_size,
    const IntArrayRef dilation_size,
    Dtype* data_col,
    // this function initializes the input data into a form that is suitable for matrix multiplication
    // dim = 2 means 2d convolution and dim = 3 means 3d convolution
    bool is_channels_last = false) {
  if (dim == 3) {
    vol2col<Dtype>(
        data_hvol,
        channels,
        input_size[0],
        input_size[1],
        input_size[2],
        output_size[0],
        output_size[1],
        output_size[2],
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
        pad_size[0],
        pad_size[1],
        pad_size[2],
        stride_size[0],
        stride_size[1],
        stride_size[2],
        dilation_size[0],
        dilation_size[1],
        dilation_size[2],
        data_col);
  }
  if (dim == 2) {
    im2col<Dtype>(
        data_hvol,
        channels,
        input_size[0],
        input_size[1],
        output_size[0],
        output_size[1],
        kernel_size[0],
        kernel_size[1],
        pad_size[0],
        pad_size[1],
        stride_size[0],
        stride_size[1],
        dilation_size[0],
        dilation_size[1],
        data_col,
        is_channels_last);
  }
}

// convert the columns back to n-dimensional volume
template <typename Dtype, int64_t dim>
void col2hvol(
    const Dtype* data_col,
    const int channels,
    const IntArrayRef input_size,
    const IntArrayRef output_size,
    const IntArrayRef kernel_size,
    const IntArrayRef stride_size,
    const IntArrayRef pad_size,
    const IntArrayRef dilation_size,
    Dtype* data_hvol,
    // rearrange the column data back into the original volume format
    // used during the backward pass for computing the gradients
    bool is_channels_last = false) {
  if (dim == 3) {
    col2vol<Dtype>(
        data_col,
        channels,
        input_size[0],
        input_size[1],
        input_size[2],
        output_size[0],
        output_size[1],
        output_size[2],
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
        pad_size[0],
        pad_size[1],
        pad_size[2],
        stride_size[0],
        stride_size[1],
        stride_size[2],
        dilation_size[0],
        dilation_size[1],
        dilation_size[2],
        data_hvol);
  }
  if (dim == 2) {
    col2im<Dtype>(
        data_col,
        channels,
        input_size[0],
        input_size[1],
        output_size[0],
        output_size[1],
        kernel_size[0],
        kernel_size[1],
        pad_size[0],
        pad_size[1],
        stride_size[0],
        stride_size[1],
        dilation_size[0],
        dilation_size[1],
        data_hvol,
        is_channels_last);
  }
}

// check the location of the tensor data fed to this function
void slow_conv_dilated_location_check(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& grad_output) {
  checkBackend("slow_conv_dilated_location_check", {input, weight}, Backend::CPU);
  if (bias.defined()) {
    checkBackend("slow_conv_dilated_location_check", {bias}, Backend::CPU);
  }
  if (grad_output.defined()) {
    checkBackend("slow_conv_dilated_location_check", {grad_output}, Backend::CPU);
  }
  // this function does not check for the location of other tensor arguments as they are stored in the same data location as the input tensor
}

// main worker function for computing the slow dilated convolution on CPU
// it computes the tensors output, grad_input, grad_weight, and/or grad_bias if defined, respectively.
template <int64_t dim>
void slow_conv_dilated_all_cpu_template(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& grad_output,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size,
    bool is_channels_last = false) {
        
    // check tensor locations
    slow_conv_dilated_location_check(input, weight, bias, grad_output);
    auto options = input.options();
    // calculate the input size
    auto input_size = input.sizes().slice(2);
    // calculate the output size
    auto output_size = internal::get_output_size<dim>(
        input, kernel_size, stride_size, pad_size, dilation_size);
    // get the batch size and the number of input/output planes
    int64_t batchSize = input.size(0);
    int64_t nInputPlane = weight.size(1);
    int64_t nOutputPlane = weight.size(0);
    Tensor columns = at::empty({0}, options);
    if (output.defined() || grad_weight.defined() || grad_input.defined()) {
        const int64_t m = c10::multiply_integers(kernel_size);
        const int64_t n = c10::multiply_integers(output_size);
        if (is_channels_last) {
        columns.resize_({n, m * nInputPlane}); // resize the columns tensor
        } else {
        columns.resize_({nInputPlane * m, n}); // resize the columns tensor
        }
    }
    // initialize the gradients if required
    if (grad_weight.defined()) {
        grad_weight.zero_();
    }
    if (grad_bias.defined()) {
        grad_bias.zero_();
    }
    if (output.defined() && !bias.defined()) {
        output.zero_();
    }
    // helpers
    Tensor grad_output_n;
    std::vector<int64_t> dims(dim);
    std::iota(dims.begin(), dims.end(), 1);

    // this is the main computation loop
    AT_DISPATCH_FLOATING_TYPES_AND3(
        at::ScalarType::Long, at::ScalarType::BFloat16, at::ScalarType::Half, input.scalar_type(), "slow_conv_dilated<>", [&] {
    // iterate over each element in the batch
    for (const auto elt : c10::irange(batchSize)) {
      // Matrix multiply per output:
      Tensor input_n = input.select(0, elt);

      // forward pass
      if (output.defined()) {
        Tensor output_n = output.select(0, elt);
        // apply the bias if defined
        if (bias.defined()) {
          /*
            (this comment is from the github repo, not replacing as I would have to closely follow it for explaining)
            Compute:

              output_n = bias * ones^T

            where

              bias is viewed as bias.view(nOutputPlane, 1)

              ones is viewed as ones.view(outputHeight * outputWidth, 1)

              output_n is viewed as output_n.view(nOutputPlane, outputHeight
          * outputWidth)

          gemm assumes column-major matrices:

            output_n^T = ones * bias^T
            C = alpha * op(A) * op(B)
            op(A) = 't', op(B) = 'n', alpha=1, beta=0
          */
          // The following for-loop is equivalent to the above
          // gemm setup but avoids allocation of ones tensor:
          for (const auto n : c10::irange(nOutputPlane)) {
            output_n.select(0, n).fill_(bias[n]);
          }
        }
        // extract the columns
        hvol2col<scalar_t, dim>(
            input_n.const_data_ptr<scalar_t>(),
            nInputPlane,
            input_size,
            output_size,
            kernel_size,
            stride_size,
            pad_size,
            dilation_size,
            columns.mutable_data_ptr<scalar_t>(),
            is_channels_last);
        /*
          (this comment is from the github repo, not replacing as I would have to closely follow it for explaining)
          Compute:

            output_n = weight * columns + output_n

          where

            weight is viewed as weight.view(nOutputPlane, nInputPlane * kD *
          kH * kW)

            columns size is (nInputPlane * kH * kW) x (outputHeight *
          outputWidth)

            output_n is viewed as output_n.view(nOutputPlane, outputHeight *
          outputWidth)

          gemm assumes column-major matrices:

          channels last:
            output_n^T = weight *columns^T + output_n^T
            C = alpha * op(A) * op(B) + beta * C
            op(A) = 't', op(B) = 'n', alpha=1, beta=1

          channels first:
            output_n^T = columns^T * weight^T + output_n^T
            C = alpha * op(A) * op(B) + beta * C
            op(A) = 'n', op(B) = 'n', alpha=1, beta=1
        */

        if (is_channels_last) {
          cpublas::gemm(
              /*transa=*/TransposeType::Transpose,
              /*transb=*/TransposeType::NoTranspose,
              /*     m=*/nOutputPlane,
              /*     n=*/columns.size(0),
              /*     k=*/columns.size(1),
              /* alpha=*/static_cast<scalar_t>(1),
              /*     A=*/weight.const_data_ptr<scalar_t>(),
              /*   lda=*/columns.size(1),
              /*     B=*/columns.const_data_ptr<scalar_t>(),
              /*   lda=*/columns.size(1),
              /*  beta=*/static_cast<scalar_t>(1),
              /*     C=*/output_n.mutable_data_ptr<scalar_t>(),
              /*   ldc=*/nOutputPlane);
        } else {
          cpublas::gemm(
              /*transa=*/TransposeType::NoTranspose,
              /*transb=*/TransposeType::NoTranspose,
              /*     m=*/columns.size(1),
              /*     n=*/nOutputPlane,
              /*     k=*/columns.size(0),
              /* alpha=*/static_cast<scalar_t>(1),
              /*     A=*/columns.const_data_ptr<scalar_t>(),
              /*   lda=*/columns.size(1),
              /*     B=*/weight.const_data_ptr<scalar_t>(),
              /*   ldb=*/columns.size(0),
              /*  beta=*/static_cast<scalar_t>(1),
              /*     C=*/output_n.mutable_data_ptr<scalar_t>(),
              /*   ldc=*/columns.size(1));
        }
      } else {
        // get all the gradients from the forward pass
        grad_output_n = grad_output.select(0, elt);
      }

      // backward pass
      if (grad_input.defined()) {
        /*
          Compute:

            columns = weight^T * grad_output_n

          where

            weight is viewed as weight.view(nOutputPlane, nInputPlane * kH *
          kW)

            grad_output_n is viewed as grad_output_n.view(nOutputPlane,
          outputHeight * outputWidth)

            columns size is (nInputPlane * kH * kW) x (outputHeight *
          outputWidth)

          gemm assumes column-major matrices:

          channels last:
            columns^T = weight^T * grad_output_n^T
            C = alpha * op(A) * op(B) + beta * C
            op(A) = 'n', op(B) = 'n', alpha=1, beta=0

          channels first:
            columns^T = grad_output_n^T * weight
            C = alpha * op(A) * op(B) + beta * C
            op(A) = 'n', op(B) = 't', alpha=1, beta=0
         */
        if (is_channels_last) {
          cpublas::gemm(
              /*transa=*/TransposeType::NoTranspose,
              /*transb=*/TransposeType::NoTranspose,
              /*     m=*/columns.size(1),
              /*     n=*/columns.size(0),
              /*     k=*/nOutputPlane,
              /* alpha=*/static_cast<scalar_t>(1),
              /*     A=*/weight.const_data_ptr<scalar_t>(),
              /*   lda=*/columns.size(1),
              /*     B=*/grad_output_n.const_data_ptr<scalar_t>(),
              /*   ldb=*/nOutputPlane,
              /*  beta=*/static_cast<scalar_t>(0),
              /*     C=*/columns.mutable_data_ptr<scalar_t>(),
              /*   ldc=*/columns.size(1));
        } else {
          cpublas::gemm(
              /*transa=*/TransposeType::NoTranspose,
              /*transb=*/TransposeType::Transpose,
              /*     m=*/columns.size(1),
              /*     n=*/columns.size(0),
              /*     k=*/nOutputPlane,
              /* alpha=*/static_cast<scalar_t>(1),
              /*     A=*/grad_output_n.const_data_ptr<scalar_t>(),
              /*   lda=*/columns.size(1),
              /*     B=*/weight.const_data_ptr<scalar_t>(),
              /*   ldb=*/columns.size(0),
              /*  beta=*/static_cast<scalar_t>(0),
              /*     C=*/columns.mutable_data_ptr<scalar_t>(),
              /*   ldc=*/columns.size(1));
        }
        // Unpack columns back into input:
        Tensor grad_input_n = grad_input.select(0, elt);

        // convert the columns back to input volume form
        col2hvol<scalar_t, dim>(
            columns.data_ptr<scalar_t>(),
            nInputPlane,
            input_size,
            output_size,
            kernel_size,
            stride_size,
            pad_size,
            dilation_size,
            grad_input_n.data_ptr<scalar_t>(),
            is_channels_last);
      }

      // compute the gradient of the weight:
      if (grad_weight.defined()) {
        // extract columns:
        hvol2col<scalar_t, dim>(
            input_n.const_data_ptr<scalar_t>(),
            nInputPlane,
            input_size,
            output_size,
            kernel_size,
            stride_size,
            pad_size,
            dilation_size,
            columns.mutable_data_ptr<scalar_t>(),
            is_channels_last);
        scalar_t scale = 1; // TODO: expose as argument?
        /*
          (this comment is from the github repo, not replacing as I would have to closely follow it for explaining)
          Compute:

            grad_weight = scale * grad_output_n * columns^T + grad_weight

          where

            grad_output_n is viewed as grad_output_n.view(nOutputPlane,
          outputHeight * outputWidth)

            columns size is (nInputPlane * kD * kH * kW) x (outputHeight *
          outputWidth)

            grad_weight is viewed as grad_weight.view(nOutputPlane,
          nInputPlane * kH * kW)

          gemm assumes column-major matrices:

          channels last:
            grad_weight^T = scale * columns^T * grad_output_n + grad_weight^T
            C = alpha * op(A) * op(B) + beta * C
            op(A) = 'n', op(B) = 't', alpha=scale, beta=1

          channels first:
            grad_weight^T = scale * columns * grad_output_n^T + grad_weight^T
            C = alpha * op(A) * op(B) + beta * C
            op(A) = 't', op(B) = 'n', alpha=scale, beta=1
        */
        if (is_channels_last) {
          cpublas::gemm(
              /*transa=*/TransposeType::NoTranspose,
              /*transb=*/TransposeType::Transpose,
              /*     m=*/columns.size(1),
              /*     n=*/nOutputPlane,
              /*     k=*/columns.size(0),
              /* alpha=*/static_cast<scalar_t>(scale),
              /*     A=*/columns.const_data_ptr<scalar_t>(),
              /*   lda=*/columns.size(1),
              /*     B=*/grad_output_n.const_data_ptr<scalar_t>(),
              /*   ldb=*/nOutputPlane,
              /*  beta=*/static_cast<scalar_t>(1),
              /*     C=*/grad_weight.mutable_data_ptr<scalar_t>(),
              /*   ldc=*/columns.size(1));
        } else {
          cpublas::gemm(
              /*transa=*/TransposeType::Transpose,
              /*transb=*/TransposeType::NoTranspose,
              /*     m=*/columns.size(0),
              /*     n=*/nOutputPlane,
              /*     k=*/columns.size(1),
              /* alpha=*/static_cast<scalar_t>(scale),
              /*     A=*/columns.const_data_ptr<scalar_t>(),
              /*   lda=*/columns.size(1),
              /*     B=*/grad_output_n.const_data_ptr<scalar_t>(),
              /*   ldb=*/columns.size(1),
              /*  beta=*/static_cast<scalar_t>(1),
              /*     C=*/grad_weight.mutable_data_ptr<scalar_t>(),
              /*   ldc=*/columns.size(0));
        }
      }

      // compute the gradient of the bias:
      if (grad_bias.defined()) {
        /*
        (this comment is from the github repo, not replacing as I would have to closely follow it for explaining)
          Compute:
            grad_bias = scale * grad_output_n * ones + grad_bias

          where

            grad_bias is viewed as grad_bias.view(nOutputPlane, 1)

            ones is viewed as ones.view(outputHeight * outputWidth, 1)

            grad_output_n is viewed as grad_output_n.view(nOutputPlane,
          outputHeight * outputWidth)

          gemm assumes column-major matrices:

            grad_bias^T = scale * grad_output_n * ones + grad_bias^T
            y = alpha * op(A) * x + beta * y
            op(A) = 't', alpha=scale, beta=1
         */
        // The following expression is equivalent to the above
        // gemm setup but avoids allocation of ones tensor:
        // sum the grad_output across the spatial dimensions
        grad_bias += grad_output_n.sum(dims);
        /*
          TODO: when scale != 1 is introduced then use:
            grad_bias += scale * grad_output_n.sum(dims);
         */
      }
    }
  });

} // slow_conv_dilated_all_cpu_template

} // namespace

// forward pass for 2d dilated convolution
Tensor slow_conv_dilated2d_cpu(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size, const std::optional<Tensor>& bias_opt,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
    // checks on the input function and tensor preparation
    // See [Note: hacky wrapper removal for optional tensor]
    c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const Tensor& bias = *bias_maybe_owned;

    bool use_channels_last = thnn_conv_use_channels_last(input, weight);
    auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

    Tensor undefined;
    internal::slow_conv_dilated_shape_check<2>(
        input,
        weight,
        bias,
        undefined,
        kernel_size,
        stride_size,
        pad_size,
        dilation_size);
    auto is_batch = input.dim() == 4;
    auto options = input.options();
    // calculate the output tensor size
    auto output_size = internal::get_output_size<2>(
        input, weight, kernel_size, stride_size, pad_size, dilation_size);
    // template function assumes batched tensors.  unsqueeze(0) will
    // insert batch dimension without affecting the original tensor.
    const Tensor input_ =
        (is_batch ? input.contiguous(memory_format) : input.contiguous().unsqueeze(0));
    const Tensor weight_ = weight.contiguous(memory_format);
    const Tensor bias_ = (bias.defined() ? bias.contiguous() : undefined);
    Tensor output = at::empty(output_size, options.memory_format(memory_format));
    Tensor output_ = (is_batch ? output : output.unsqueeze(0));

    // call the main worker function
    slow_conv_dilated_all_cpu_template<2>(
        output_,
        input_,
        weight_,
        bias_,
        undefined,
        undefined,
        undefined,
        undefined,
        kernel_size,
        stride_size,
        pad_size,
        dilation_size,
        use_channels_last);
    return output;
}

// forward pass for 2d dilated convolution
Tensor slow_conv_dilated3d_cpu(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size, const std::optional<Tensor>& bias_opt,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
    // checks on the input function and tensor preparation
    // See [Note: hacky wrapper removal for optional tensor]
    c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const Tensor& bias = *bias_maybe_owned;

    Tensor undefined;
    internal::slow_conv_dilated_shape_check<3>(
        input,
        weight,
        bias,
        undefined,
        kernel_size,
        stride_size,
        pad_size,
        dilation_size);
    auto is_batch = input.dim() == 5;
    auto options = input.options();
    // calculate the output tensor size
    auto output_size = internal::get_output_size<3>(
        input, weight, kernel_size, stride_size, pad_size, dilation_size);
    // template function assumes batched tensors.  unsqueeze(0) will
    // insert batch dimension without affecting the original tensor.
    const Tensor input_ =
        (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
    const Tensor weight_ = weight.contiguous();
    const Tensor bias_ = (bias.defined() ? bias.contiguous() : undefined);
    Tensor output = at::empty(output_size, options);
    Tensor output_ = (is_batch ? output : output.unsqueeze(0));

    // call the main worker function
    slow_conv_dilated_all_cpu_template<3>(
        output,
        input_,
        weight_,
        bias_,
        undefined,
        undefined,
        undefined,
        undefined,
        kernel_size,
        stride_size,
        pad_size,
        dilation_size);
  return output;
}

// backward pass for 2d dilated convolution
static std::tuple<Tensor, Tensor, Tensor> slow_conv_dilated2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size,
    const std::array<bool, 3ul> output_mask) {
    bool use_channels_last = thnn_conv_use_channels_last(input, weight);
    auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

    Tensor undefined;
    internal::slow_conv_dilated_shape_check<2>(
        input,
        weight,
        undefined,
        grad_output,
        kernel_size,
        stride_size,
        pad_size,
        dilation_size);
    auto is_batch = input.dim() == 4;
    auto options = grad_output.options();
    // template function assumes batched tensors.  unsqueeze(0) will
    // insert batch dimension without affecting the original tensor.
    const Tensor grad_output_ =
        (is_batch ? grad_output.contiguous(memory_format)
                    : grad_output.contiguous().unsqueeze(0));
    const Tensor input_ =
        (is_batch ? input.contiguous(memory_format) : input.contiguous().unsqueeze(0));
    const Tensor weight_ = weight.contiguous(memory_format);
    // only compute gradients which have their corresponding output_mask as true:
    Tensor grad_input =
        (output_mask[0] ? at::empty(input.sizes(), options.memory_format(memory_format)) : undefined);
    Tensor grad_weight =
        (output_mask[1] ? at::empty(weight.sizes(), options.memory_format(memory_format)) : undefined);
    Tensor grad_bias =
        (output_mask[2] ? at::empty(weight.size(0), options) : undefined);
    Tensor grad_input_ =
        (output_mask[0] ? (is_batch ? grad_input : grad_input.unsqueeze(0))
                        : undefined);
    // call the main worker function
    slow_conv_dilated_all_cpu_template<2>(
        undefined,
        input_,
        weight_,
        undefined,
        grad_output_,
        grad_input,
        grad_weight,
        grad_bias,
        kernel_size,
        stride_size,
        pad_size,
        dilation_size,
        use_channels_last);
  return std::tie(grad_input, grad_weight, grad_bias);
}

// backward pass for 3d dilated convolution
static std::tuple<Tensor, Tensor, Tensor> slow_conv_dilated3d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size,
    const std::array<bool, 3ul> output_mask) {
    Tensor undefined;
    internal::slow_conv_dilated_shape_check<3>(
        input,
        weight,
        undefined,
        grad_output,
        kernel_size,
        stride_size,
        pad_size,
        dilation_size);
    auto is_batch = input.dim() == 5;
    auto options = grad_output.options();
    // template function assumes batched tensors.  unsqueeze(0) will
    // insert batch dimension without affecting the original tensor.
    const Tensor grad_output_ =
        (is_batch ? grad_output.contiguous()
                    : grad_output.contiguous().unsqueeze(0));
    const Tensor input_ =
        (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
    const Tensor weight_ = weight.contiguous();
    // compute only gradients for which the corresponding output_mask is true:
    Tensor grad_input =
        (output_mask[0] ? at::empty(input.sizes(), options) : undefined);
    Tensor grad_weight =
        (output_mask[1] ? at::empty(weight.sizes(), options) : undefined);
    Tensor grad_bias =
        (output_mask[2] ? at::empty(weight.size(0), options) : undefined);
    Tensor grad_input_ =
        (output_mask[0] ? (is_batch ? grad_input : grad_input.unsqueeze(0))
                        : undefined);
    // call the main worker function
    slow_conv_dilated_all_cpu_template<3>(
        undefined,
        input_,
        weight_,
        undefined,
        grad_output_,
        grad_input,
        grad_weight,
        grad_bias,
        kernel_size,
        stride_size,
        pad_size,
        dilation_size);
  return std::tie(grad_input, grad_weight, grad_bias);
}

// register the CPU dispatch for backward passes
REGISTER_ALL_CPU_DISPATCH(slow_conv_dilated2d_backward_stub, &slow_conv_dilated2d_backward_cpu);
REGISTER_ALL_CPU_DISPATCH(slow_conv_dilated3d_backward_stub, &slow_conv_dilated3d_backward_cpu);

} // namespace at::native