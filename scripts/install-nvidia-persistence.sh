#!/usr/bin/env bash
# Install a systemd service to keep nvidia-persistenced enabled across reboots.
# This prevents cudaErrorLaunchTimeout on GPUs that also drive a display.
#
# Run once: sudo ./scripts/install-nvidia-persistence.sh

set -euo pipefail

cat > /etc/systemd/system/nvidia-persistence.service << 'EOF'
[Unit]
Description=NVIDIA Persistence Mode
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi --persistence-mode=1
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now nvidia-persistence.service
echo "Done. nvidia-smi persistence mode will now survive reboots."
