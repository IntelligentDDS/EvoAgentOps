import psutil
import csv
import threading
import os
import time
from datetime import datetime


class CustomResourceMonitor:
    def __init__(self, output_dir="./monitoring_data/metrics", sample_interval=1):
        self.output_dir = output_dir
        self.sample_interval = sample_interval
        self.running = True
        self.proc = psutil.Process()
        self.current_process = self.proc
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(output_dir, f"{timestamp}.csv")

        # Pre-build all metric functions - build once to avoid repeated computation
        self.headers, self.funcs = self._build_all_metrics()
        self._init_csv()

        # Start daemon thread
        threading.Thread(target=self._loop, daemon=True).start()

    def _get_cached_system_data(self):
        """Get all system data at once and cache it"""
        cached_data = {}

        # Get all potentially repeated system data calls at once
        try:
            cached_data["cpu_times"] = psutil.cpu_times()
        except:
            cached_data["cpu_times"] = None

        try:
            cached_data["cpu_stats"] = psutil.cpu_stats()
        except:
            cached_data["cpu_stats"] = None

        try:
            cached_data["virtual_memory"] = psutil.virtual_memory()
        except:
            cached_data["virtual_memory"] = None

        try:
            cached_data["swap_memory"] = psutil.swap_memory()
        except:
            cached_data["swap_memory"] = None

        try:
            cached_data["disk_io_counters"] = psutil.disk_io_counters()
        except:
            cached_data["disk_io_counters"] = None

        try:
            cached_data["net_io_counters"] = psutil.net_io_counters()
        except:
            cached_data["net_io_counters"] = None

        try:
            cached_data["net_connections"] = psutil.net_connections() if hasattr(psutil, "net_connections") else None
        except:
            cached_data["net_connections"] = None

        try:
            cached_data["cpu_freq"] = psutil.cpu_freq() if hasattr(psutil, "cpu_freq") else None
        except:
            cached_data["cpu_freq"] = None

        try:
            cached_data["getloadavg"] = psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
        except:
            cached_data["getloadavg"] = None

        try:
            cached_data["sensors_battery"] = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
        except:
            cached_data["sensors_battery"] = None

        # Key optimization: get temperature sensor data at once
        try:
            cached_data["sensors_temperatures"] = (
                psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else None
            )
        except:
            cached_data["sensors_temperatures"] = None

        try:
            cached_data["sensors_fans"] = psutil.sensors_fans() if hasattr(psutil, "sensors_fans") else None
        except:
            cached_data["sensors_fans"] = None

        try:
            cached_data["net_if_addrs"] = psutil.net_if_addrs() if hasattr(psutil, "net_if_addrs") else None
        except:
            cached_data["net_if_addrs"] = None

        return cached_data

    def _get_cached_process_data(self):
        """Get all process data at once and cache it"""
        cached_data = {}

        try:
            cached_data["cpu_times"] = self.current_process.cpu_times()
        except:
            cached_data["cpu_times"] = None

        try:
            cached_data["memory_info"] = self.current_process.memory_info()
        except:
            cached_data["memory_info"] = None

        try:
            cached_data["memory_full_info"] = self.current_process.memory_full_info()
        except:
            cached_data["memory_full_info"] = None

        try:
            cached_data["num_ctx_switches"] = self.current_process.num_ctx_switches()
        except:
            cached_data["num_ctx_switches"] = None

        try:
            cached_data["io_counters"] = (
                self.current_process.io_counters() if hasattr(self.current_process, "io_counters") else None
            )
        except:
            cached_data["io_counters"] = None

        try:
            cached_data["net_connections"] = self.current_process.net_connections()
        except:
            cached_data["net_connections"] = []

        try:
            cached_data["open_files"] = self.current_process.open_files()
        except:
            cached_data["open_files"] = []

        try:
            cached_data["children"] = self.current_process.children()
        except:
            cached_data["children"] = []

        try:
            cached_data["children_recursive"] = self.current_process.children(recursive=True)
        except:
            cached_data["children_recursive"] = []

        return cached_data

    def _build_all_metrics(self):
        """Build comprehensive metrics architecture"""
        headers = ["timestamp"]
        funcs = [lambda: datetime.utcnow().isoformat()]

        # Build a template first, actual data is fetched in the loop
        template_metrics = self._get_template_metrics()

        for name in template_metrics:
            headers.append(name)
            # Create placeholder functions, actual execution is handled in _loop
            funcs.append(lambda name=name: None)

        return headers, funcs

    def _get_template_metrics(self):
        """Get all metric name templates"""
        template = [
            # CPU core metrics
            "sys_cpu_count_logical",
            "sys_cpu_count_physical",
            "sys_cpu_percent",
            "sys_cpu_times_user",
            "sys_cpu_times_system",
            "sys_cpu_times_idle",
            "sys_cpu_times_nice",
            "sys_cpu_times_iowait",
            "sys_cpu_times_irq",
            "sys_cpu_times_softirq",
            "sys_cpu_times_steal",
            "sys_cpu_times_guest",
            "sys_cpu_times_guest_nice",
            # CPU statistics
            "sys_cpu_ctx_switches",
            "sys_cpu_interrupts",
            "sys_cpu_soft_interrupts",
            "sys_cpu_syscalls",
            # Memory core metrics
            "sys_memory_total",
            "sys_memory_available",
            "sys_memory_percent",
            "sys_memory_used",
            "sys_memory_free",
            "sys_memory_active",
            "sys_memory_inactive",
            "sys_memory_buffers",
            "sys_memory_cached",
            "sys_memory_shared",
            "sys_memory_slab",
            "sys_memory_wired",
            # Swap memory
            "sys_swap_total",
            "sys_swap_used",
            "sys_swap_free",
            "sys_swap_percent",
            "sys_swap_sin",
            "sys_swap_sout",
            # Disk IO extended metrics
            "sys_disk_io_read_count",
            "sys_disk_io_write_count",
            "sys_disk_io_read_bytes",
            "sys_disk_io_write_bytes",
            "sys_disk_io_read_time",
            "sys_disk_io_write_time",
            "sys_disk_io_busy_time",
            "sys_disk_io_read_merged_count",
            "sys_disk_io_write_merged_count",
            # Network IO extended metrics
            "sys_net_io_bytes_sent",
            "sys_net_io_bytes_recv",
            "sys_net_io_packets_sent",
            "sys_net_io_packets_recv",
            "sys_net_io_errin",
            "sys_net_io_errout",
            "sys_net_io_dropin",
            "sys_net_io_dropout",
            # System load and status metrics
            "sys_boot_time",
            "sys_users_count",
            "sys_process_count",
            "sys_uptime_seconds",
            # Network connection statistics
            "sys_net_connections_count",
            "sys_net_connections_tcp",
            "sys_net_connections_udp",
            "sys_net_connections_unix",
            "sys_net_connections_inet",
            "sys_net_connections_inet4",
            "sys_net_connections_inet6",
            # Disk partition info
            "sys_disk_partitions_count",
            # Process basic info
            "proc_pid",
            "proc_ppid",
            "proc_name",
            "proc_status",
            "proc_create_time",
            "proc_username",
            "proc_exe",
            "proc_cwd",
            # Process CPU usage extended
            "proc_cpu_percent",
            "proc_cpu_times_user",
            "proc_cpu_times_system",
            "proc_cpu_times_children_user",
            "proc_cpu_times_children_system",
            "proc_cpu_num",
            "proc_num_ctx_switches_voluntary",
            "proc_num_ctx_switches_involuntary",
            # Process memory usage extended
            "proc_memory_rss",
            "proc_memory_vms",
            "proc_memory_shared",
            "proc_memory_text",
            "proc_memory_lib",
            "proc_memory_data",
            "proc_memory_dirty",
            "proc_memory_percent",
            # Process memory extended info
            "proc_memory_uss",
            "proc_memory_pss",
            "proc_memory_swap",
            # Process priority and resource limits
            "proc_nice",
            "proc_ionice_class",
            "proc_ionice_value",
            # Process threads and resources extended
            "proc_num_threads",
            "proc_num_fds",
            "proc_num_handles",
            # Process IO statistics extended
            "proc_io_read_count",
            "proc_io_write_count",
            "proc_io_read_bytes",
            "proc_io_write_bytes",
            "proc_io_read_chars",
            "proc_io_write_chars",
            # Process network connections
            "proc_connections_count",
            "proc_connections_tcp",
            "proc_connections_udp",
            # Process open files
            "proc_open_files_count",
            # Process children
            "proc_children_count",
            "proc_children_recursive_count",
            # Process status
            "proc_is_running",
            # Process environment variables count
            "proc_environ_count",
            # Process thread details
            "proc_threads_count",
            # Process memory maps
            "proc_memory_maps_count",
        ]

        # Dynamically add optional metrics
        # CPU frequency info
        if hasattr(psutil, "cpu_freq"):
            try:
                freq = psutil.cpu_freq()
                if freq:
                    template.extend(["sys_cpu_freq_current", "sys_cpu_freq_min", "sys_cpu_freq_max"])
            except:
                pass

        # System load average (Unix systems)
        if hasattr(psutil, "getloadavg"):
            try:
                load = psutil.getloadavg()
                if load:
                    template.extend(["sys_loadavg_1min", "sys_loadavg_5min", "sys_loadavg_15min"])
            except:
                pass

        # Battery info
        if hasattr(psutil, "sensors_battery"):
            try:
                battery = psutil.sensors_battery()
                if battery:
                    template.extend(["sys_battery_percent", "sys_battery_secsleft", "sys_battery_power_plugged"])
            except:
                pass

        # Temperature sensors (Linux/macOS)
        if hasattr(psutil, "sensors_temperatures"):
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if ("cpu" in name.lower() or "core" in name.lower()) and entries:
                            clean_name = name.replace("-", "_").replace(" ", "_")
                            template.extend(
                                [
                                    f"sys_temp_{clean_name}_current",
                                    f"sys_temp_{clean_name}_high",
                                    f"sys_temp_{clean_name}_critical",
                                ]
                            )
                            break
            except:
                pass

        # Fan info (Linux)
        if hasattr(psutil, "sensors_fans"):
            try:
                fans = psutil.sensors_fans()
                if fans:
                    template.append("sys_fans_count")
            except:
                pass

        # Network interface statistics
        if hasattr(psutil, "net_if_addrs"):
            try:
                net_if_addrs = psutil.net_if_addrs()
                if net_if_addrs:
                    template.append("sys_net_interfaces_count")
            except:
                pass

        return template

    def _build_metrics_with_cache(self, sys_cache, proc_cache):
        """Build metric functions using cached data"""
        funcs = [lambda: datetime.utcnow().isoformat()]

        # System metrics
        sys_metrics = {
            # CPU core metrics
            "sys_cpu_count_logical": lambda: psutil.cpu_count(logical=True),
            "sys_cpu_count_physical": lambda: psutil.cpu_count(logical=False),
            "sys_cpu_percent": lambda: psutil.cpu_percent(interval=None),
            "sys_cpu_times_user": lambda: sys_cache["cpu_times"].user if sys_cache["cpu_times"] else None,
            "sys_cpu_times_system": lambda: sys_cache["cpu_times"].system if sys_cache["cpu_times"] else None,
            "sys_cpu_times_idle": lambda: sys_cache["cpu_times"].idle if sys_cache["cpu_times"] else None,
            "sys_cpu_times_nice": lambda: (
                getattr(sys_cache["cpu_times"], "nice", None) if sys_cache["cpu_times"] else None
            ),
            "sys_cpu_times_iowait": lambda: (
                getattr(sys_cache["cpu_times"], "iowait", None) if sys_cache["cpu_times"] else None
            ),
            "sys_cpu_times_irq": lambda: (
                getattr(sys_cache["cpu_times"], "irq", None) if sys_cache["cpu_times"] else None
            ),
            "sys_cpu_times_softirq": lambda: (
                getattr(sys_cache["cpu_times"], "softirq", None) if sys_cache["cpu_times"] else None
            ),
            "sys_cpu_times_steal": lambda: (
                getattr(sys_cache["cpu_times"], "steal", None) if sys_cache["cpu_times"] else None
            ),
            "sys_cpu_times_guest": lambda: (
                getattr(sys_cache["cpu_times"], "guest", None) if sys_cache["cpu_times"] else None
            ),
            "sys_cpu_times_guest_nice": lambda: (
                getattr(sys_cache["cpu_times"], "guest_nice", None) if sys_cache["cpu_times"] else None
            ),
            # CPU statistics
            "sys_cpu_ctx_switches": lambda: (sys_cache["cpu_stats"].ctx_switches if sys_cache["cpu_stats"] else None),
            "sys_cpu_interrupts": lambda: (sys_cache["cpu_stats"].interrupts if sys_cache["cpu_stats"] else None),
            "sys_cpu_soft_interrupts": lambda: (
                sys_cache["cpu_stats"].soft_interrupts if sys_cache["cpu_stats"] else None
            ),
            "sys_cpu_syscalls": lambda: sys_cache["cpu_stats"].syscalls if sys_cache["cpu_stats"] else None,
            # Memory core metrics
            "sys_memory_total": lambda: (sys_cache["virtual_memory"].total if sys_cache["virtual_memory"] else None),
            "sys_memory_available": lambda: (
                sys_cache["virtual_memory"].available if sys_cache["virtual_memory"] else None
            ),
            "sys_memory_percent": lambda: (
                sys_cache["virtual_memory"].percent if sys_cache["virtual_memory"] else None
            ),
            "sys_memory_used": lambda: (sys_cache["virtual_memory"].used if sys_cache["virtual_memory"] else None),
            "sys_memory_free": lambda: (sys_cache["virtual_memory"].free if sys_cache["virtual_memory"] else None),
            "sys_memory_active": lambda: (
                getattr(sys_cache["virtual_memory"], "active", None) if sys_cache["virtual_memory"] else None
            ),
            "sys_memory_inactive": lambda: (
                getattr(sys_cache["virtual_memory"], "inactive", None) if sys_cache["virtual_memory"] else None
            ),
            "sys_memory_buffers": lambda: (
                getattr(sys_cache["virtual_memory"], "buffers", None) if sys_cache["virtual_memory"] else None
            ),
            "sys_memory_cached": lambda: (
                getattr(sys_cache["virtual_memory"], "cached", None) if sys_cache["virtual_memory"] else None
            ),
            "sys_memory_shared": lambda: (
                getattr(sys_cache["virtual_memory"], "shared", None) if sys_cache["virtual_memory"] else None
            ),
            "sys_memory_slab": lambda: (
                getattr(sys_cache["virtual_memory"], "slab", None) if sys_cache["virtual_memory"] else None
            ),
            "sys_memory_wired": lambda: (
                getattr(sys_cache["virtual_memory"], "wired", None) if sys_cache["virtual_memory"] else None
            ),
            # Swap memory
            "sys_swap_total": lambda: sys_cache["swap_memory"].total if sys_cache["swap_memory"] else None,
            "sys_swap_used": lambda: sys_cache["swap_memory"].used if sys_cache["swap_memory"] else None,
            "sys_swap_free": lambda: sys_cache["swap_memory"].free if sys_cache["swap_memory"] else None,
            "sys_swap_percent": lambda: (sys_cache["swap_memory"].percent if sys_cache["swap_memory"] else None),
            "sys_swap_sin": lambda: sys_cache["swap_memory"].sin if sys_cache["swap_memory"] else None,
            "sys_swap_sout": lambda: sys_cache["swap_memory"].sout if sys_cache["swap_memory"] else None,
            # Disk IO extended metrics
            "sys_disk_io_read_count": lambda: (
                sys_cache["disk_io_counters"].read_count if sys_cache["disk_io_counters"] else None
            ),
            "sys_disk_io_write_count": lambda: (
                sys_cache["disk_io_counters"].write_count if sys_cache["disk_io_counters"] else None
            ),
            "sys_disk_io_read_bytes": lambda: (
                sys_cache["disk_io_counters"].read_bytes if sys_cache["disk_io_counters"] else None
            ),
            "sys_disk_io_write_bytes": lambda: (
                sys_cache["disk_io_counters"].write_bytes if sys_cache["disk_io_counters"] else None
            ),
            "sys_disk_io_read_time": lambda: (
                sys_cache["disk_io_counters"].read_time if sys_cache["disk_io_counters"] else None
            ),
            "sys_disk_io_write_time": lambda: (
                sys_cache["disk_io_counters"].write_time if sys_cache["disk_io_counters"] else None
            ),
            "sys_disk_io_busy_time": lambda: (
                getattr(sys_cache["disk_io_counters"], "busy_time", None) if sys_cache["disk_io_counters"] else None
            ),
            "sys_disk_io_read_merged_count": lambda: (
                getattr(sys_cache["disk_io_counters"], "read_merged_count", None)
                if sys_cache["disk_io_counters"]
                else None
            ),
            "sys_disk_io_write_merged_count": lambda: (
                getattr(sys_cache["disk_io_counters"], "write_merged_count", None)
                if sys_cache["disk_io_counters"]
                else None
            ),
            # Network IO extended metrics
            "sys_net_io_bytes_sent": lambda: (
                sys_cache["net_io_counters"].bytes_sent if sys_cache["net_io_counters"] else None
            ),
            "sys_net_io_bytes_recv": lambda: (
                sys_cache["net_io_counters"].bytes_recv if sys_cache["net_io_counters"] else None
            ),
            "sys_net_io_packets_sent": lambda: (
                sys_cache["net_io_counters"].packets_sent if sys_cache["net_io_counters"] else None
            ),
            "sys_net_io_packets_recv": lambda: (
                sys_cache["net_io_counters"].packets_recv if sys_cache["net_io_counters"] else None
            ),
            "sys_net_io_errin": lambda: (sys_cache["net_io_counters"].errin if sys_cache["net_io_counters"] else None),
            "sys_net_io_errout": lambda: (
                sys_cache["net_io_counters"].errout if sys_cache["net_io_counters"] else None
            ),
            "sys_net_io_dropin": lambda: (
                sys_cache["net_io_counters"].dropin if sys_cache["net_io_counters"] else None
            ),
            "sys_net_io_dropout": lambda: (
                sys_cache["net_io_counters"].dropout if sys_cache["net_io_counters"] else None
            ),
            # System load and status metrics
            "sys_boot_time": lambda: psutil.boot_time(),
            "sys_users_count": lambda: len(psutil.users()),
            "sys_process_count": lambda: len(psutil.pids()),
            "sys_uptime_seconds": lambda: time.time() - psutil.boot_time(),
            # Network connection statistics
            "sys_net_connections_count": lambda: (
                len(sys_cache["net_connections"]) if sys_cache["net_connections"] else None
            ),
            "sys_net_connections_tcp": lambda: (
                len([c for c in sys_cache["net_connections"] if c.type == psutil.SOCK_STREAM])
                if sys_cache["net_connections"]
                else None
            ),
            "sys_net_connections_udp": lambda: (
                len([c for c in sys_cache["net_connections"] if c.type == psutil.SOCK_DGRAM])
                if sys_cache["net_connections"]
                else None
            ),
            "sys_net_connections_unix": lambda: (
                len([c for c in sys_cache["net_connections"] if c.family == psutil.AF_UNIX])
                if sys_cache["net_connections"]
                else None
            ),
            "sys_net_connections_inet": lambda: (
                len([c for c in sys_cache["net_connections"] if c.family in (psutil.AF_INET, psutil.AF_INET6)])
                if sys_cache["net_connections"]
                else None
            ),
            "sys_net_connections_inet4": lambda: (
                len([c for c in sys_cache["net_connections"] if c.family == psutil.AF_INET])
                if sys_cache["net_connections"]
                else None
            ),
            "sys_net_connections_inet6": lambda: (
                len([c for c in sys_cache["net_connections"] if c.family == psutil.AF_INET6])
                if sys_cache["net_connections"]
                else None
            ),
            # Disk partition info
            "sys_disk_partitions_count": lambda: len(psutil.disk_partitions()),
        }

        # Process metrics
        proc_metrics = {
            # Basic info
            "proc_pid": lambda: self.current_process.pid,
            "proc_ppid": lambda: self.current_process.ppid(),
            "proc_name": lambda: self.current_process.name(),
            "proc_status": lambda: self.current_process.status(),
            "proc_create_time": lambda: self.current_process.create_time(),
            "proc_username": lambda: self.current_process.username(),
            "proc_exe": lambda: self.current_process.exe(),
            "proc_cwd": lambda: self.current_process.cwd(),
            # CPU usage extended
            "proc_cpu_percent": lambda: self.current_process.cpu_percent(interval=None),
            "proc_cpu_times_user": lambda: proc_cache["cpu_times"].user if proc_cache["cpu_times"] else None,
            "proc_cpu_times_system": lambda: (proc_cache["cpu_times"].system if proc_cache["cpu_times"] else None),
            "proc_cpu_times_children_user": lambda: (
                proc_cache["cpu_times"].children_user if proc_cache["cpu_times"] else None
            ),
            "proc_cpu_times_children_system": lambda: (
                proc_cache["cpu_times"].children_system if proc_cache["cpu_times"] else None
            ),
            "proc_cpu_num": lambda: getattr(self.current_process, "cpu_num", lambda: None)(),
            "proc_num_ctx_switches_voluntary": lambda: (
                proc_cache["num_ctx_switches"].voluntary if proc_cache["num_ctx_switches"] else None
            ),
            "proc_num_ctx_switches_involuntary": lambda: (
                proc_cache["num_ctx_switches"].involuntary if proc_cache["num_ctx_switches"] else None
            ),
            # Memory usage extended
            "proc_memory_rss": lambda: proc_cache["memory_info"].rss if proc_cache["memory_info"] else None,
            "proc_memory_vms": lambda: proc_cache["memory_info"].vms if proc_cache["memory_info"] else None,
            "proc_memory_shared": lambda: (
                getattr(proc_cache["memory_info"], "shared", None) if proc_cache["memory_info"] else None
            ),
            "proc_memory_text": lambda: (
                getattr(proc_cache["memory_info"], "text", None) if proc_cache["memory_info"] else None
            ),
            "proc_memory_lib": lambda: (
                getattr(proc_cache["memory_info"], "lib", None) if proc_cache["memory_info"] else None
            ),
            "proc_memory_data": lambda: (
                getattr(proc_cache["memory_info"], "data", None) if proc_cache["memory_info"] else None
            ),
            "proc_memory_dirty": lambda: (
                getattr(proc_cache["memory_info"], "dirty", None) if proc_cache["memory_info"] else None
            ),
            "proc_memory_percent": lambda: self.current_process.memory_percent(),
            # Memory extended info
            "proc_memory_uss": lambda: (
                getattr(proc_cache["memory_full_info"], "uss", None) if proc_cache["memory_full_info"] else None
            ),
            "proc_memory_pss": lambda: (
                getattr(proc_cache["memory_full_info"], "pss", None) if proc_cache["memory_full_info"] else None
            ),
            "proc_memory_swap": lambda: (
                getattr(proc_cache["memory_full_info"], "swap", None) if proc_cache["memory_full_info"] else None
            ),
            # Process priority and resource limits
            "proc_nice": lambda: self.current_process.nice(),
            "proc_ionice_class": lambda: (
                self.current_process.ionice().ioclass if hasattr(self.current_process, "ionice") else None
            ),
            "proc_ionice_value": lambda: (
                self.current_process.ionice().value if hasattr(self.current_process, "ionice") else None
            ),
            # Threads and resources extended
            "proc_num_threads": lambda: self.current_process.num_threads(),
            "proc_num_fds": lambda: (
                self.current_process.num_fds() if hasattr(self.current_process, "num_fds") else None
            ),
            "proc_num_handles": lambda: (self.current_process.num_handles() if psutil.WINDOWS else None),
            # IO statistics extended
            "proc_io_read_count": lambda: (proc_cache["io_counters"].read_count if proc_cache["io_counters"] else None),
            "proc_io_write_count": lambda: (
                proc_cache["io_counters"].write_count if proc_cache["io_counters"] else None
            ),
            "proc_io_read_bytes": lambda: (proc_cache["io_counters"].read_bytes if proc_cache["io_counters"] else None),
            "proc_io_write_bytes": lambda: (
                proc_cache["io_counters"].write_bytes if proc_cache["io_counters"] else None
            ),
            "proc_io_read_chars": lambda: (
                proc_cache["io_counters"].read_chars
                if proc_cache["io_counters"] and hasattr(proc_cache["io_counters"], "read_chars")
                else None
            ),
            "proc_io_write_chars": lambda: (
                proc_cache["io_counters"].write_chars
                if proc_cache["io_counters"] and hasattr(proc_cache["io_counters"], "write_chars")
                else None
            ),
            # Network connections
            "proc_connections_count": lambda: len(proc_cache["net_connections"]),
            "proc_connections_tcp": lambda: len(
                [c for c in proc_cache["net_connections"] if c.type == psutil.SOCK_STREAM]
            ),
            "proc_connections_udp": lambda: len(
                [c for c in proc_cache["net_connections"] if c.type == psutil.SOCK_DGRAM]
            ),
            # Open files
            "proc_open_files_count": lambda: len(proc_cache["open_files"]),
            # Children
            "proc_children_count": lambda: len(proc_cache["children"]),
            "proc_children_recursive_count": lambda: len(proc_cache["children_recursive"]),
            # Process status
            "proc_is_running": lambda: int(self.current_process.is_running()),
            # Process environment variables count
            "proc_environ_count": lambda: (
                len(self.current_process.environ()) if hasattr(self.current_process, "environ") else None
            ),
            # Process thread details
            "proc_threads_count": lambda: (
                len(self.current_process.threads()) if hasattr(self.current_process, "threads") else None
            ),
            # Process memory maps
            "proc_memory_maps_count": lambda: (
                len(self.current_process.memory_maps()) if hasattr(self.current_process, "memory_maps") else None
            ),
        }

        # Add optional advanced metrics
        # CPU frequency info
        if sys_cache["cpu_freq"]:
            sys_metrics.update(
                {
                    "sys_cpu_freq_current": lambda: sys_cache["cpu_freq"].current,
                    "sys_cpu_freq_min": lambda: sys_cache["cpu_freq"].min,
                    "sys_cpu_freq_max": lambda: sys_cache["cpu_freq"].max,
                }
            )

        # System load average (Unix systems)
        if sys_cache["getloadavg"]:
            sys_metrics.update(
                {
                    "sys_loadavg_1min": lambda: sys_cache["getloadavg"][0],
                    "sys_loadavg_5min": lambda: sys_cache["getloadavg"][1],
                    "sys_loadavg_15min": lambda: sys_cache["getloadavg"][2],
                }
            )

        # Battery info
        if sys_cache["sensors_battery"]:
            sys_metrics.update(
                {
                    "sys_battery_percent": lambda: sys_cache["sensors_battery"].percent,
                    "sys_battery_secsleft": lambda: sys_cache["sensors_battery"].secsleft,
                    "sys_battery_power_plugged": lambda: int(sys_cache["sensors_battery"].power_plugged),
                }
            )

        # Temperature sensors (Linux/macOS) - Key optimization point!
        if sys_cache["sensors_temperatures"]:
            for name, entries in sys_cache["sensors_temperatures"].items():
                if ("cpu" in name.lower() or "core" in name.lower()) and entries:
                    clean_name = name.replace("-", "_").replace(" ", "_")
                    sensor_data = entries[0]  # Cache sensor data
                    # Use closure to capture sensor data, avoid repeated calls
                    sys_metrics[f"sys_temp_{clean_name}_current"] = lambda data=sensor_data: data.current
                    sys_metrics[f"sys_temp_{clean_name}_high"] = lambda data=sensor_data: (
                        data.high if data.high else None
                    )
                    sys_metrics[f"sys_temp_{clean_name}_critical"] = lambda data=sensor_data: (
                        data.critical if data.critical else None
                    )
                    break

        # Fan info (Linux)
        if sys_cache["sensors_fans"]:
            sys_metrics["sys_fans_count"] = lambda: sum(len(entries) for entries in sys_cache["sensors_fans"].values())

        # Network interface statistics
        if sys_cache["net_if_addrs"]:
            sys_metrics["sys_net_interfaces_count"] = lambda: len(sys_cache["net_if_addrs"])

        # Merge all metrics and sort by headers order
        all_metrics = {**sys_metrics, **proc_metrics}

        # Add functions in headers order
        for header in self.headers[1:]:  # Skip timestamp
            if header in all_metrics:
                funcs.append(all_metrics[header])
            else:
                funcs.append(lambda: None)  # Placeholder

        return funcs

    def _init_csv(self):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.output_file, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(self.headers)

    def _loop(self):
        """Monitor main loop - async write to avoid blocking"""
        while self.running:
            # Re-fetch cached data each loop, then execute all functions
            sys_cache = self._get_cached_system_data()
            proc_cache = self._get_cached_process_data()

            # Rebuild all metric functions to use new cached data
            current_funcs = self._build_metrics_with_cache(sys_cache, proc_cache)

            data = []
            for func in current_funcs:
                try:
                    value = func()
                    data.append(value)
                except Exception as e:
                    data.append(None)

            # Async write to CSV
            threading.Thread(target=self._write_csv, args=(data,), daemon=True).start()
            threading.Event().wait(self.sample_interval)

    def _write_csv(self, data):
        """Async write to CSV"""
        try:
            with open(self.output_file, "a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(data)
        except Exception as e:
            print(f"Error writing to CSV: {e}")

    def stop(self):
        """Stop monitoring"""
        self.running = False


# Usage example
# if __name__ == "__main__":
#     monitor = CustomResourceMonitor()
#     try:
#         # Let monitoring run for a while
#         time.sleep(30)
#     except KeyboardInterrupt:
#         print("Stopping monitoring...")
#     finally:
#         monitor.stop()
