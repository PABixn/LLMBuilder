from __future__ import annotations

from .schemas import RunPodGpuCatalogItem, RunPodProviderCatalog


_RUNPOD_GPU_CATALOG: tuple[RunPodGpuCatalogItem, ...] = (
    RunPodGpuCatalogItem(id="NVIDIA GeForce RTX 3070", display_name="RTX 3070", memory_gb=8),
    RunPodGpuCatalogItem(id="NVIDIA GeForce RTX 3080", display_name="RTX 3080", memory_gb=10),
    RunPodGpuCatalogItem(id="NVIDIA GeForce RTX 3080 Ti", display_name="RTX 3080 Ti", memory_gb=12),
    RunPodGpuCatalogItem(id="NVIDIA GeForce RTX 4070 Ti", display_name="RTX 4070 Ti", memory_gb=12),
    RunPodGpuCatalogItem(id="NVIDIA GeForce RTX 4080", display_name="RTX 4080", memory_gb=16),
    RunPodGpuCatalogItem(id="NVIDIA GeForce RTX 4080 SUPER", display_name="RTX 4080 SUPER", memory_gb=16),
    RunPodGpuCatalogItem(id="NVIDIA GeForce RTX 5080", display_name="RTX 5080", memory_gb=16),
    RunPodGpuCatalogItem(id="NVIDIA RTX 2000 Ada Generation", display_name="RTX 2000 Ada", memory_gb=None),
    RunPodGpuCatalogItem(id="NVIDIA RTX 4000 Ada Generation", display_name="RTX 4000 Ada", memory_gb=None),
    RunPodGpuCatalogItem(id="NVIDIA RTX A4000", display_name="RTX A4000", memory_gb=None),
    RunPodGpuCatalogItem(id="Tesla V100-PCIE-16GB", display_name="Tesla V100 PCIe", memory_gb=None),
    RunPodGpuCatalogItem(id="NVIDIA A30", display_name="A30", memory_gb=24),
    RunPodGpuCatalogItem(id="NVIDIA GeForce RTX 3090", display_name="RTX 3090", memory_gb=24),
    RunPodGpuCatalogItem(id="NVIDIA GeForce RTX 3090 Ti", display_name="RTX 3090 Ti", memory_gb=24),
    RunPodGpuCatalogItem(id="NVIDIA GeForce RTX 4090", display_name="RTX 4090", memory_gb=24),
    RunPodGpuCatalogItem(id="NVIDIA L4", display_name="L4", memory_gb=24),
    RunPodGpuCatalogItem(id="NVIDIA RTX A4500", display_name="RTX A4500", memory_gb=None),
    RunPodGpuCatalogItem(id="NVIDIA RTX A5000", display_name="RTX A5000", memory_gb=None),
    RunPodGpuCatalogItem(id="NVIDIA GeForce RTX 5090", display_name="RTX 5090", memory_gb=32),
    RunPodGpuCatalogItem(id="NVIDIA RTX 5000 Ada Generation", display_name="RTX 5000 Ada", memory_gb=None),
    RunPodGpuCatalogItem(id="Tesla V100-SXM2-32GB", display_name="Tesla V100 SXM2", memory_gb=None),
    RunPodGpuCatalogItem(id="NVIDIA A40", display_name="A40", memory_gb=48),
    RunPodGpuCatalogItem(id="NVIDIA L40", display_name="L40", memory_gb=48),
    RunPodGpuCatalogItem(id="NVIDIA L40S", display_name="L40S", memory_gb=48),
    RunPodGpuCatalogItem(id="NVIDIA RTX 6000 Ada Generation", display_name="RTX 6000 Ada", memory_gb=None),
    RunPodGpuCatalogItem(id="NVIDIA RTX A6000", display_name="RTX A6000", memory_gb=None),
    RunPodGpuCatalogItem(id="NVIDIA A100 80GB PCIe", display_name="A100 PCIe", memory_gb=80),
    RunPodGpuCatalogItem(id="NVIDIA A100-SXM4-80GB", display_name="A100 SXM", memory_gb=80),
    RunPodGpuCatalogItem(id="NVIDIA H100 80GB HBM3", display_name="H100 SXM", memory_gb=80),
    RunPodGpuCatalogItem(id="NVIDIA H100 PCIe", display_name="H100 PCIe", memory_gb=80),
    RunPodGpuCatalogItem(id="NVIDIA H100 NVL", display_name="H100 NVL", memory_gb=94),
    RunPodGpuCatalogItem(
        id="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        display_name="RTX PRO 6000 Blackwell Server",
        memory_gb=None,
    ),
    RunPodGpuCatalogItem(
        id="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        display_name="RTX PRO 6000 Blackwell Workstation",
        memory_gb=None,
    ),
    RunPodGpuCatalogItem(id="NVIDIA H200", display_name="H200 SXM", memory_gb=141),
    RunPodGpuCatalogItem(id="NVIDIA H200 NVL", display_name="H200 NVL", memory_gb=143),
    RunPodGpuCatalogItem(id="NVIDIA B200", display_name="B200", memory_gb=180),
    RunPodGpuCatalogItem(id="AMD Instinct MI300X OAM", display_name="MI300X", memory_gb=192),
    RunPodGpuCatalogItem(id="NVIDIA B300 SXM6 AC", display_name="B300", memory_gb=288),
)


def build_runpod_provider_catalog(default_gpu_type_id: str) -> RunPodProviderCatalog:
    catalog_items = list(_RUNPOD_GPU_CATALOG)
    if default_gpu_type_id and all(item.id != default_gpu_type_id for item in catalog_items):
        catalog_items.append(
            RunPodGpuCatalogItem(
                id=default_gpu_type_id,
                display_name=default_gpu_type_id,
                memory_gb=None,
            )
        )
    return RunPodProviderCatalog(gpu_options=catalog_items)
