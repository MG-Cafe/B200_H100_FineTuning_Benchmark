"""
Simulated CrowdStrike Malware Analysis Training Data Generator

Simulates the multimodal fine-tuning dataset described in the meeting:
- Input 1: Raw bytes from compiled executables (~512KB binary blobs)
- Input 2: Short natural language questions/instructions
- Output: Short text descriptions of behavioral traits
- Labels: benign/malicious classification

The data is saved in a format ready for fine-tuning (JSON Lines).
"""

import json
import os
import random
import struct
from pathlib import Path

# --- Configuration ---
NUM_SAMPLES = 500
OUTPUT_DIR = "simulated_training_data"
OUTPUT_FILE = "train.jsonl"
MAX_BINARY_SIZE_BYTES = 512 * 1024  # 512 KB
MIN_BINARY_SIZE_BYTES = 64 * 1024   # 64 KB

# --- Templates ---

QUESTIONS = [
    "What are the behavioral traits you expect to see from this sample?",
    "Analyze this binary and describe its likely functionality.",
    "What system-level behaviors does this executable exhibit?",
    "Describe the potential malicious indicators in this binary.",
    "What capabilities does this program appear to have?",
    "Identify the key functional characteristics of this executable.",
    "What network and system behaviors does this sample demonstrate?",
    "Provide a behavioral summary of this binary executable.",
    "What are the suspicious and benign traits observed in this sample?",
    "Describe what this executable is likely designed to do.",
    "Assess the threat level and functionality of this binary.",
    "What API calls and system interactions does this binary suggest?",
    "Summarize the behavioral profile of this executable sample.",
    "What indicators of compromise can be inferred from this binary?",
    "Evaluate the risk and behavioral characteristics of this program.",
]

SYSTEM_CALLS = [
    "NtCreateFile", "NtWriteFile", "NtReadFile", "NtClose",
    "CreateProcessW", "VirtualAllocEx", "WriteProcessMemory",
    "NtQuerySystemInformation", "RegOpenKeyExW", "RegSetValueExW",
    "WSAStartup", "connect", "send", "recv", "InternetOpenA",
    "HttpOpenRequestA", "HttpSendRequestA", "URLDownloadToFileA",
    "CreateRemoteThread", "LoadLibraryA", "GetProcAddress",
    "SetWindowsHookExA", "OpenProcess", "ReadProcessMemory",
    "CryptEncrypt", "CryptDecrypt", "CryptHashData",
    "NtMapViewOfSection", "NtUnmapViewOfSection",
    "AdjustTokenPrivileges", "LookupPrivilegeValueA",
    "CreateServiceA", "StartServiceA", "ChangeServiceConfigA",
    "NtDelayExecution", "GetTickCount", "QueryPerformanceCounter",
    "IsDebuggerPresent", "CheckRemoteDebuggerPresent",
    "GetSystemInfo", "GlobalMemoryStatusEx",
    "FindFirstFileW", "FindNextFileW", "DeleteFileW", "MoveFileW",
    "CreateMutexA", "WaitForSingleObject",
    "ShellExecuteA", "WinExec", "system",
]

BEHAVIORS = [
    "attempts outbound network connections on port {port}",
    "resolves DNS queries to suspicious domains",
    "modifies Windows registry run keys for persistence",
    "creates a scheduled task for periodic execution",
    "injects code into remote process memory space",
    "spawns child processes with elevated privileges",
    "reads and exfiltrates browser credential stores",
    "encrypts files on disk using AES-256 encryption",
    "establishes a reverse shell connection to {ip}",
    "downloads additional payloads from remote C2 server",
    "hooks keyboard input for keystroke logging",
    "disables Windows Defender real-time protection",
    "creates a Windows service for persistence",
    "performs process hollowing on legitimate system binaries",
    "scans local network for SMB shares on port 445",
    "exfiltrates data via DNS tunneling",
    "detects virtual machine environments and alters behavior",
    "checks for debugger presence using IsDebuggerPresent",
    "uses reflective DLL injection techniques",
    "communicates with C2 using HTTPS with custom certificates",
    "drops files into the Windows Temp directory",
    "modifies the hosts file to redirect traffic",
    "enumerates running processes to identify security tools",
    "uses process doppelganging to evade detection",
    "performs time-based evasion with delayed execution",
    "accesses cryptocurrency wallet files",
    "captures screenshots at regular intervals",
    "records audio through the system microphone",
    "propagates via removable USB drives",
    "exploits known CVE vulnerabilities for privilege escalation",
    "contains encrypted configuration data in the .data section",
    "uses steganography to hide data in image files",
    "patches AMSI (Antimalware Scan Interface) in memory",
    "performs credential dumping from LSASS process memory",
    "utilizes living-off-the-land binaries (LOLBins)",
]

BENIGN_BEHAVIORS = [
    "standard CRT initialization routines are present",
    "appears to be a legitimate software installer",
    "contains valid digital signature from known vendor",
    "performs standard file I/O operations",
    "uses standard Windows API for GUI rendering",
    "contains typical application update check mechanisms",
    "loads standard system DLLs (kernel32, user32, advapi32)",
    "performs expected configuration file read/write operations",
    "has a standard PE header with valid checksum",
    "contains legitimate logging and telemetry code",
    "includes standard error handling and crash reporting",
    "uses documented COM interfaces for system interaction",
]

MALWARE_FAMILIES = [
    "Emotet", "TrickBot", "Ryuk", "Cobalt Strike", "Mimikatz",
    "AgentTesla", "FormBook", "LokiBot", "NanoCore", "RemcosRAT",
    "QakBot", "IcedID", "Dridex", "ZLoader", "BazarLoader",
    "Conti", "REvil", "DarkSide", "BlackMatter", "LockBit",
    "AsyncRAT", "njRAT", "DcRAT", "RedLineStealer", "Raccoon",
    "generic trojan", "generic backdoor", "generic dropper",
    "generic ransomware", "generic infostealer", "generic RAT",
    "generic worm", "generic cryptominer", "generic rootkit",
]

FILE_TYPES = [
    "PE32 executable", "PE32+ executable (x86-64)",
    "DLL (Dynamic Link Library)", ".NET assembly",
    "packed executable (UPX)", "packed executable (Themida)",
    "packed executable (VMProtect)", "self-extracting archive",
    "Windows installer (MSI)", "batch script dropper",
]


def generate_realistic_binary(size: int) -> bytes:
    """
    Generate a binary blob that loosely resembles a PE executable.
    Includes a DOS header stub, PE signature, and then random/structured data.
    """
    parts = []

    # DOS Header (MZ)
    dos_header = bytearray(64)
    dos_header[0:2] = b"MZ"
    dos_header[2:4] = struct.pack("<H", random.randint(0, 511))
    dos_header[60:64] = struct.pack("<I", 64)  # PE header offset
    parts.append(bytes(dos_header))

    # PE Signature
    parts.append(b"PE\x00\x00")

    # COFF Header (20 bytes)
    coff = bytearray(20)
    coff[0:2] = struct.pack("<H", random.choice([0x14C, 0x8664]))  # i386 or AMD64
    coff[2:4] = struct.pack("<H", random.randint(1, 10))  # number of sections
    coff[4:8] = struct.pack("<I", random.randint(1600000000, 1740000000))  # timestamp
    coff[16:18] = struct.pack("<H", random.choice([0x0002, 0x0020, 0x0102]))  # characteristics
    parts.append(bytes(coff))

    # Some section name stubs
    section_names = [b".text\x00\x00\x00", b".rdata\x00\x00", b".data\x00\x00\x00",
                     b".rsrc\x00\x00\x00", b".reloc\x00\x00", b".pdata\x00\x00"]
    for name in random.sample(section_names, min(len(section_names), random.randint(2, 5))):
        parts.append(name)
        parts.append(os.urandom(32))  # section header data

    # Some string table entries (simulating embedded strings)
    strings = [
        b"kernel32.dll\x00", b"ntdll.dll\x00", b"user32.dll\x00",
        b"advapi32.dll\x00", b"ws2_32.dll\x00", b"wininet.dll\x00",
        b"GetProcAddress\x00", b"LoadLibraryA\x00", b"VirtualAlloc\x00",
        b"CreateFileW\x00", b"WriteFile\x00", b"ReadFile\x00",
        b"RegOpenKeyExW\x00", b"InternetOpenA\x00", b"HttpSendRequestA\x00",
        b"CreateProcessW\x00", b"VirtualProtectEx\x00",
        b"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\x00",
        b"http://update.microsoft.com\x00",
        b"Mozilla/5.0 (Windows NT 10.0; Win64; x64)\x00",
        b"%TEMP%\\svchost.exe\x00",
        b"cmd.exe /c ping 127.0.0.1\x00",
    ]

    current_size = sum(len(p) for p in parts)

    # Fill remaining with mix of random bytes and embedded strings
    while current_size < size:
        if random.random() < 0.15 and strings:
            s = random.choice(strings)
            parts.append(s)
            current_size += len(s)
        else:
            # Random chunk (simulating code/data sections)
            chunk_size = min(random.randint(256, 4096), size - current_size)
            if chunk_size <= 0:
                break
            parts.append(os.urandom(chunk_size))
            current_size += chunk_size

    raw = b"".join(parts)
    return raw[:size]


def generate_response(is_malicious: bool) -> str:
    """Generate a realistic short analyst response about a binary."""
    parts = []

    if is_malicious:
        # File type
        file_type = random.choice(FILE_TYPES)
        parts.append(f"This sample is a {file_type}.")

        # Malware family (sometimes)
        if random.random() < 0.6:
            family = random.choice(MALWARE_FAMILIES)
            confidence = random.choice(["high", "moderate", "low"])
            parts.append(
                f"Based on behavioral analysis, this binary exhibits characteristics "
                f"consistent with the {family} family ({confidence} confidence)."
            )

        # System calls observed
        num_syscalls = random.randint(3, 8)
        calls = random.sample(SYSTEM_CALLS, num_syscalls)
        parts.append(
            f"The executable makes native system calls including {', '.join(calls)}."
        )

        # Malicious behaviors
        num_behaviors = random.randint(2, 6)
        selected_behaviors = random.sample(BEHAVIORS, num_behaviors)
        formatted = []
        for b in selected_behaviors:
            b = b.format(
                port=random.choice([80, 443, 445, 4444, 8080, 8443, 1337, 9001]),
                ip=f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
            )
            formatted.append(b)
        parts.append("Key behavioral indicators: " + "; ".join(formatted) + ".")

        # Encrypted strings
        if random.random() < 0.5:
            parts.append(
                "The binary contains encrypted or obfuscated strings suggesting "
                "an attempt to evade static analysis."
            )

        # Risk assessment
        risk = random.choice(["critical", "high", "elevated"])
        parts.append(f"Overall threat assessment: {risk} risk. Recommended action: quarantine and investigate.")

    else:
        file_type = random.choice(FILE_TYPES[:4])  # more likely legit types
        parts.append(f"This sample is a {file_type}.")

        # Benign behaviors
        num_benign = random.randint(2, 5)
        selected = random.sample(BENIGN_BEHAVIORS, num_benign)
        parts.append("Observed characteristics: " + "; ".join(selected) + ".")

        # Some system calls (benign ones)
        benign_calls = ["CreateFileW", "ReadFile", "WriteFile", "RegOpenKeyExW",
                        "GetSystemInfo", "LoadLibraryA", "GetProcAddress",
                        "FindFirstFileW", "FindNextFileW", "NtClose"]
        num_calls = random.randint(2, 5)
        calls = random.sample(benign_calls, num_calls)
        parts.append(f"System calls observed: {', '.join(calls)}.")

        # Verdict
        parts.append(
            "No malicious indicators detected. This appears to be a legitimate "
            "application performing expected operations."
        )
        parts.append("Overall threat assessment: benign. No action required.")

    return " ".join(parts)


def binary_to_hex_tokens(binary_data: bytes, max_tokens: int = 65000) -> str:
    """
    Convert raw binary to a hex string representation (simulating tokenization).
    """
    hex_str = binary_data.hex()
    max_chars = max_tokens * 2  # rough approximation
    return hex_str[:max_chars]


def generate_sample(sample_id: int) -> dict:
    """Generate a single training sample."""
    # Determine if malicious
    is_malicious = random.random() < 0.5

    # Generate binary
    binary_size = random.randint(MIN_BINARY_SIZE_BYTES, MAX_BINARY_SIZE_BYTES)
    binary_data = generate_realistic_binary(binary_size)

    # Convert binary to hex token representation
    binary_hex = binary_to_hex_tokens(binary_data)

    # Select a question
    question = random.choice(QUESTIONS)

    # Generate response
    response = generate_response(is_malicious)

    # Build the training sample in chat/instruction format
    sample = {
        "id": f"sim_{sample_id:06d}",
        "label": "malicious" if is_malicious else "benign",
        "binary_size_bytes": binary_size,
        "binary_hex_length": len(binary_hex),
        "conversations": [
            {
                "role": "system",
                "content": (
                    "You are a malware analyst AI. You are given raw bytes from a "
                    "compiled executable program and a question about it. Analyze the "
                    "binary and provide a detailed behavioral assessment."
                )
            },
            {
                "role": "user",
                "content": f"<binary>{binary_hex}</binary>\n\n{question}"
            },
            {
                "role": "assistant",
                "content": response
            }
        ],
        # Metadata for benchmarking
        "metadata": {
            "input_tokens_approx": len(binary_hex) // 4 + len(question.split()),
            "output_tokens_approx": len(response.split()),
            "is_malicious": is_malicious,
            "binary_size_kb": binary_size / 1024,
        }
    }

    return sample


def main():
    print(f"Generating {NUM_SAMPLES} simulated training samples...")
    print(f"Binary size range: {MIN_BINARY_SIZE_BYTES // 1024}KB - {MAX_BINARY_SIZE_BYTES // 1024}KB")
    print(f"Output directory: {OUTPUT_DIR}/")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    total_binary_bytes = 0
    malicious_count = 0
    benign_count = 0

    with open(output_path, "w") as f:
        for i in range(NUM_SAMPLES):
            sample = generate_sample(i)
            f.write(json.dumps(sample) + "\n")

            total_binary_bytes += sample["binary_size_bytes"]
            if sample["label"] == "malicious":
                malicious_count += 1
            else:
                benign_count += 1

            if (i + 1) % 50 == 0:
                print(f" Generated {i + 1}/{NUM_SAMPLES} samples...")

    # Dataset statistics
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"Dataset generation complete!")
    print(f"{'=' * 60}")
    print(f"Output file: {output_path}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Total samples: {NUM_SAMPLES}")
    print(f"Malicious samples: {malicious_count}")
    print(f"Benign samples: {benign_count}")
    print(f"Total binary data: {total_binary_bytes / (1024 * 1024):.1f} MB")
    print(f"Avg binary size: {total_binary_bytes / NUM_SAMPLES / 1024:.1f} KB")
    print(f"{'=' * 60}")

    # Print a sample for verification
    with open(output_path, "r") as f:
        first_sample = json.loads(f.readline())

    print(f"\nSample record (truncated binary):")
    print(f" ID: {first_sample['id']}")
    print(f" Label: {first_sample['label']}")
    print(f" Binary size: {first_sample['binary_size_bytes']} bytes")
    print(f" Hex length: {first_sample['binary_hex_length']} chars")
    print(f" Question: {first_sample['conversations'][1]['content'][:80]}...")
    print(f" Response: {first_sample['conversations'][2]['content'][:120]}...")
    print(f" Input tokens (approx): {first_sample['metadata']['input_tokens_approx']}")
    print(f" Output tokens (approx): {first_sample['metadata']['output_tokens_approx']}")


if __name__ == "__main__":
    main()
