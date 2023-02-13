/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// CUDA Library include(s).
#include "../utils/utils.hpp"
#include "traccc/cuda/utils/definitions.hpp"
#include "traccc/cuda/clusterization/clusterization_cca_algorithm.hpp"
#include "traccc/cuda/clusterization/clusterization_kernels.cuh"

// Project include(s)
#include "traccc/clusterization/device/form_spacepoints.hpp"
#include "traccc/cuda/cca/component_connection.hpp"
#include "traccc/cuda/utils/make_prefix_sum_buff.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"

// Vecmem include(s).
#include <vecmem/utils/copy.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// System include(s).
#include <algorithm>

// Local include(s)
#include "traccc/cuda/utils/definitions.hpp"

namespace traccc::cuda {

clusterization_cca_algorithm::clusterization_cca_algorithm(
    const traccc::memory_resource& mr, stream& str)
    : m_mr(mr), m_stream(str) {

    // Initialize m_copy ptr based on memory resources that were given
    if (mr.host) {
        m_copy = std::make_unique<vecmem::cuda::copy>();
    } else {
        m_copy = std::make_unique<vecmem::copy>();
    }
}

clusterization_cca_algorithm::output_type clusterization_cca_algorithm::operator()(
    const cell_container_types::host& cells_per_event) const {

    // Vecmem copy object for moving the data between host and device
    vecmem::copy copy;

    // Initialize the device container for cells
    cell_container_types::host cells_device(cells_per_event);

    // Initialize the host container for cells
    //cell_container_types::host cells_host(cells_device);

    // Number of modules
    unsigned int num_modules = cells_per_event.size();
    
    // Work block size for kernel execution
    std::size_t threadsPerBlock = 64;
    
    traccc::cuda::component_connection cc;

    traccc::measurement_container_types::host measurements = cc(cells_per_event);
    
    const auto& measurement_data = get_data(measurements, m_mr.host ? m_mr.host : &(m_mr.main));

    traccc::measurement_container_types::const_view measurements_view(measurement_data);

    std::vector<std::size_t> clusters_per_module_host(num_modules);

    for(int i = 0; i < clusters_per_module_host.size(); i++)
    {
      clusters_per_module_host[i] = measurements_view.items.ptr()[i].size();
    }

    // Create prefix sum buffer
    vecmem::data::vector_buffer meas_prefix_sum_buff = make_prefix_sum_buff(
      m_copy->get_sizes(measurements_view.items), *m_copy, m_mr);

    /*measurement_container_types::buffer measurements_buffer{
      {num_modules, m_mr.main},
      {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
        m_mr.main, m_mr.host}};
    m_copy->setup(measurements_buffer.headers);
    m_copy->setup(measurements_buffer.items);
*/
  //  (*m_copy)(measurements_view.headers, measurements_buffer.headers);
    //(*m_copy)(measurements_view.items, vecmem::get_data(measurements_buffer.items), vecmem::copy::type::host_to_device);

    std::size_t blocksPerGrid = meas_prefix_sum_buff.size()/threadsPerBlock + 1;
    
    spacepoint_container_types::buffer spacepoints_buffer{
        {num_modules, m_mr.main},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
        m_mr.main, m_mr.host}};
    m_copy->setup(spacepoints_buffer.headers);
    m_copy->setup(spacepoints_buffer.items);

    // Invoke spacepoint formation will call form_spacepoints kernel
    traccc::cuda::kernels::form_spacepoints<<<blocksPerGrid, threadsPerBlock>>>(
      measurements_view, meas_prefix_sum_buff, spacepoints_buffer);
    
    // Check for kernel launch errors and Wait for the spacepoint formation
    // kernel to finish
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    return spacepoints_buffer;
  }
}  // namespace traccc::cuda