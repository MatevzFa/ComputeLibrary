#include <cstddef>
#include <memory>
#include <vector>

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "src/core/CL/kernels/CLGEMMMatrixMultiplyNativeKernel.h"

namespace arm_compute
{
class CLCol2ImKernel;
class CLIm2ColKernel;
class CLWeightsReshapeKernel;
class ICLTensor;

class AccumulatingGEMM : public IFunction
{
    public:
    /** Constructor
     *
     * @param[in] memory_manager  (Optional) Memory manager.
     * @param[in] weights_manager (Optional) Weights manager.
     */
    AccumulatingGEMM(std::shared_ptr<IMemoryManager> memory_manager);

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    AccumulatingGEMM(const AccumulatingGEMM &) = delete;

    /** Default move constructor */
    AccumulatingGEMM(AccumulatingGEMM &&) = default;

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    AccumulatingGEMM &operator=(const AccumulatingGEMM &) = delete;

    /** Default move assignment operator */
    AccumulatingGEMM &operator=(AccumulatingGEMM &&) = default;

    /**Default destructor */
    ~AccumulatingGEMM();

    void configure(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output,
                   const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), unsigned int num_groups = 1);

    void configure(const CLCompileContext &compile_context,
                   const ICLTensor *input, const ICLTensor *weights, ICLTensor *output,
                   const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), unsigned int num_groups = 1);

    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output,
                           const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo(),
                           const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), unsigned int num_groups = 1);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

    void run(size_t filter_perforation);

    private:
    MemoryGroup _memory_group{};

    size_t M, C, K, HW;

    CLTensor _filter_tensor;
    CLTensor _image_tensor;
    CLTensor _output_tensor;
    CLTensor _output_buffer;

    std::vector<std::unique_ptr<CLGEMMMatrixMultiplyNativeKernel>> _filter_image_gemmkernels;

    std::vector<CLTensor> _output_tensors;
};
}; // namespace arm_compute
