/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/cuda/utils/stream.hpp"

// Project include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"


// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

//// Traccc library include(s).
//#include "traccc/utils/memory_resource.hpp"

namespace traccc::cuda {

class clusterization_cca_algorithm
    : public algorithm<spacepoint_container_types::buffer(
          const cell_container_types::host&)> {

    public:
    /// Constructor for clusterization algorithm
    ///
    /// @param mr is a memory resource (device)
    clusterization_cca_algorithm(const traccc::memory_resource& mr, stream& str);

    /// Callable operator for clusterization algorithm
    ///
    /// @param cells_per_event is a container with cell modules as headers
    /// and cells as the items
    /// @return a spacepoint container (buffer) - jagged vector of spacepoints
    /// per module.
    output_type operator()(
        const cell_container_types::host& cells_per_event) const override;

    private:
    /// The memory resource(s) to use
    traccc::memory_resource m_mr;
    /// The copy object to use
    std::unique_ptr<vecmem::copy> m_copy;
    /// The CUDA stream to use
    stream& m_stream;
};

}