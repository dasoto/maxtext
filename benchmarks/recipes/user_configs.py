# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Define user specific configurations for recipes here."""

import os

import maxtext_xpk_runner as mxr
from xpk_configs import XpkClusterConfig

cluster_config = XpkClusterConfig(
    cluster_name="bodaborg-v6e-256-lcscld-c",
    project="tpu-prod-env-one-vm",
    zone="southamerica-west1-a",
    device_type="v6e-256",
)
xpk_path = "~/xpk"

user = os.environ["USER"]
region = "-".join(cluster_config.zone.split("-")[:-1])
proxy_image = (
    f"us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_proxy_server_maxtext@sha256:31bdd23a7b3276525b13a1e635dc916aab9f2ac3852b4f3f4e353c2a790bd221"
)
server_image = (
    f"us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_server_maxtext@sha256:94d107922fc2e88700192742dac7558a12e44328bfc62987717f82fb1bfff1a6"
)
colocated_python_image = f"gcr.io/cloud-tpu-multipod-dev/ksadi_sidecar_maxtext@sha256:d1d9b88214463447c15945bfd466cdb6ba0d7f1204d7ad0ac17832ad60c9c497"
runner = f"gcr.io/{cluster_config.project}/{user}_maxtext_latest:latest"
base_output_directory = f"gs://{user}-{region}/{user}"
headless = True
pathways_config = mxr.PathwaysConfig(
    server_image=server_image,
    proxy_server_image=proxy_image,
    runner_image=runner,
    colocated_python_sidecar_image=colocated_python_image,
    headless=headless,
)
headless_workload_name = f"{user[:3]}-headless"
