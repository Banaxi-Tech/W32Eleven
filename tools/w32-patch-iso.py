#!/usr/bin/env python3
"""Patch a Windows (x64) install ISO so that BIOS machines boot it through the
W32Eleven emulator: on a 32-bit CPU, the emulator runs first and boots the
unchanged Windows setup inside an emulated x86-64 PC.

usage: w32-patch-iso.py IN.iso OUT.iso [options]
       w32-patch-iso.py --in-place IMAGE.iso [options]
       w32-patch-iso.py --restore IMAGE.iso

options:
  --emulator FILE   bare-metal emulator (default: build/emulator.elf)
  --bios FILE       SeaBIOS image for the guest (default: /usr/share/qemu/bios.bin)
  --vgabios FILE    VGA BIOS for the guest (default: /usr/share/qemu/vgabios-stdvga.bin)
  --ram MIB         guest RAM (default: all the host RAM the emulator can spare)
  --args "..."      extra emulator command line options
  --no-slim         automatic install without the slim settings for slow PCs

What changes (everything else stays byte for byte, including the UDF and
ISO 9660 file systems and the UEFI boot entry, so UEFI PCs still boot Windows
natively):
  - a GRUB El Torito image (with the emulator, BIOS and VGA BIOS embedded) is
    appended after the end of the image;
  - the BIOS ("default") entry of the El Torito boot catalog points at it;
  - the original BIOS entry is saved in the appended area (--restore undoes
    the patch, then the image can be truncated back to its old size).

  - the ISO's unused first sector gets a hybrid MBR (GRUB boot sector and one
    partition entry), so the same image also boots from a USB stick.

Boot media: CD/DVD (El Torito), USB sticks and other BIOS disks (dd the image).
"""
import os, shutil, struct, subprocess, sys, tempfile

SECTOR = 2048
MAGIC = b"W32ELEVEN-PATCH1"   # header of the appended area (sector before the GRUB image)


def die(msg):
    sys.exit(f"w32-patch-iso: {msg}")


def read_at(f, off, n):
    f.seek(off)
    b = f.read(n)
    if len(b) != n:
        die(f"short read at {off:#x}")
    return b


def boot_catalog(f):
    """LBA of the El Torito boot catalog (from the boot record at sector 17)."""
    br = read_at(f, 17 * SECTOR, SECTOR)
    if br[0] != 0 or br[1:6] != b"CD001" or not br[7:30].startswith(b"EL TORITO SPECIFICATION"):
        die("no El Torito boot record: not a bootable ISO 9660 image")
    return struct.unpack_from("<I", br, 0x47)[0]


def check_validation(cat):
    if cat[0] != 1 or cat[30:32] != b"\x55\xaa":
        die("bad El Torito validation entry")
    if sum(struct.unpack_from("<16H", cat, 0)) & 0xffff:
        die("bad El Torito validation checksum")


BOOTDEV = "cd"   # GRUB device of the boot medium: "cd" (El Torito) or "$bootdev" (USB stick / disk)


def blocklist(lba, size):
    """GRUB blocklist on the boot medium: 512-byte units from its start."""
    return f"({BOOTDEV}){lba * 4}+{(size + 511) // 512}"


def build_grub(tmp, files, ram, extra, usb=False):
    """Small GRUB El Torito image with only grub.cfg embedded; the emulator and
    firmware are raw extents after the ISO, loaded through blocklists."""
    cmd = "mode=firmware cdrom=boot"
    if ram:
        cmd += f" ram={ram}"
    if extra:
        cmd += " " + extra
    cfg = os.path.join(tmp, "grub.cfg")
    with open(cfg, "w") as c:
        if usb:   # booted from a BIOS disk (USB stick): the menu reads from the drive GRUB came from
            c.write("""regexp --set=1 bootdev '^\\(([^)]*)\\)' "$cmdpath"
if [ -z "$bootdev" ]; then set bootdev=hd0; fi
""")
        c.write(f"""set timeout=5
set default=0
menuentry "Windows on W32Eleven - with this PC's hard disk (x86-64 emulator for 32-bit PCs)" {{
    echo "Loading the W32Eleven emulator..."
    multiboot {blocklist(*files["emulator"])} {cmd} hdd=ata
    module {blocklist(*files["bios"])} bios
    module {blocklist(*files["vgabios"])} vgabios
    module {blocklist(*files["unattend"])} unattend
}}
menuentry "Windows on W32Eleven - CD only, no disk access" {{
    echo "Loading the W32Eleven emulator..."
    multiboot {blocklist(*files["emulator"])} {cmd}
    module {blocklist(*files["bios"])} bios
    module {blocklist(*files["vgabios"])} vgabios
    module {blocklist(*files["unattend"])} unattend
}}
menuentry "AUTOMATIC install of Windows 11 Pro - ERASES the first hard disk without asking" {{
    echo "Loading the W32Eleven emulator..."
    multiboot {blocklist(*files["emulator"])} {cmd} hdd=ata
    module {blocklist(*files["bios"])} bios
    module {blocklist(*files["vgabios"])} vgabios
    module {blocklist(*files["unattend_auto"])} unattend
}}
menuentry "Install the W32Eleven boot loader on the hard disk (after Windows Setup)" {{
    echo "Loading the W32Eleven emulator..."
    multiboot {blocklist(*files["emulator"])} {cmd} hdd=ata install=1
    module {blocklist(*files["bios"])} bios
    module {blocklist(*files["vgabios"])} vgabios
    module {blocklist(*files["diskstub"])} diskstub
}}
menuentry "Reboot" {{
    reboot
}}
""")
    img = os.path.join(tmp, "eltorito.img")
    mods = "multiboot normal configfile echo reboot biosdisk memdisk tar vbe video video_fb" + (" part_msdos regexp test" if usb else "")
    r = subprocess.run(["grub-mkstandalone", "-O", "i386-pc" if usb else "i386-pc-eltorito", "-o", img,
                        "--install-modules=" + mods, "--modules=" + mods,
                        "--locales=", "--fonts=", "--themes=",
                        f"boot/grub/grub.cfg={cfg}"], capture_output=True, text=True)
    if r.returncode:
        die("grub-mkstandalone failed: " + r.stderr.strip())
    return open(img, "rb").read()


UNATTEND = r"""<?xml version="1.0" encoding="utf-8"?>
<unattend xmlns="urn:schemas-microsoft-com:unattend">
  <settings pass="windowsPE">
    <component name="Microsoft-Windows-Setup" processorArchitecture="amd64" publicKeyToken="31bf3856ad364e35" language="neutral" versionScope="nonSxS" xmlns:wcm="http://schemas.microsoft.com/WMIConfig/2002/State">
      <RunSynchronous>
%s
      </RunSynchronous>
    </component>
  </settings>
</unattend>
"""
BYPASS = ["BypassTPMCheck", "BypassSecureBootCheck", "BypassRAMCheck", "BypassCPUCheck", "BypassStorageCheck"]


# automatic install: Setup's GUI checks for >= 2 active processors, which no
# setting bypasses; WinPE's own tools install Windows without that check
AUTO_CMD = r"""@echo off
echo W32Eleven automatic install: ERASES disk 0
set SRC=
for %%d in (C D E F G H I J K L M N O P Q R S T U V Y Z) do if exist %%d:\sources\install.wim set SRC=%%d:\sources\install.wim
if not defined SRC for %%d in (C D E F G H I J K L M N O P Q R S T U V Y Z) do if exist %%d:\sources\install.esd set SRC=%%d:\sources\install.esd
if not defined SRC (echo install image not found & pause & exit /b 1)
set ME=%~d0
(echo select disk 0
echo clean
echo create partition primary
echo format fs=ntfs quick label=Windows
echo active
echo assign letter=W) > X:\w32dp.txt
diskpart /s X:\w32dp.txt || (echo diskpart failed & pause & exit /b 1)
dism /Apply-Image /ImageFile:%SRC% /Name:"Windows 11 Pro" /ApplyDir:W:\ || dism /Apply-Image /ImageFile:%SRC% /Index:1 /ApplyDir:W:\ || (echo dism failed & pause & exit /b 1)
%SLIM%bcdboot W:\Windows /s W: /f BIOS || (echo bcdboot failed & pause & exit /b 1)
mkdir W:\Windows\Panther
copy /y %ME%\oobe.xml W:\Windows\Panther\unattend.xml
wpeutil reboot
"""
# "Slim" settings for slow PCs, written into the applied image offline (before
# Windows first starts) by the automatic install. Hives: SYS = SYSTEM (ControlSet001),
# SW = SOFTWARE, DU = the default user profile (copied to every new account).
SLIM_REG = [
    # services that do background work: prefetch/superfetch, search indexing, telemetry,
    # maps, Defender (antivirus: real-time scanning, network inspection, boot driver)
    ("SYS", r"ControlSet001\Services\SysMain", "Start", "REG_DWORD", "4"),
    ("SYS", r"ControlSet001\Services\WSearch", "Start", "REG_DWORD", "4"),
    ("SYS", r"ControlSet001\Services\DiagTrack", "Start", "REG_DWORD", "4"),
    ("SYS", r"ControlSet001\Services\dmwappushservice", "Start", "REG_DWORD", "4"),
    ("SYS", r"ControlSet001\Services\MapsBroker", "Start", "REG_DWORD", "4"),
    ("SYS", r"ControlSet001\Services\WinDefend", "Start", "REG_DWORD", "4"),
    ("SYS", r"ControlSet001\Services\WdNisSvc", "Start", "REG_DWORD", "4"),
    ("SYS", r"ControlSet001\Services\WdNisDrv", "Start", "REG_DWORD", "4"),
    ("SYS", r"ControlSet001\Services\WdFilter", "Start", "REG_DWORD", "4"),
    ("SYS", r"ControlSet001\Services\WdBoot", "Start", "REG_DWORD", "4"),
    ("SYS", r"ControlSet001\Services\Sense", "Start", "REG_DWORD", "4"),
    # no fast startup / hibernation (would write all of RAM to disk at every shutdown)
    ("SYS", r"ControlSet001\Control\Session Manager\Power", "HiberbootEnabled", "REG_DWORD", "0"),
    ("SYS", r"ControlSet001\Control\Power", "HibernateEnabled", "REG_DWORD", "0"),
    # policies: Defender off, no telemetry, no background apps, no consumer content, no widgets,
    # no automatic Store and Edge updates, Edge does not stay running in the background
    ("SW", r"Policies\Microsoft\Windows Defender", "DisableAntiSpyware", "REG_DWORD", "1"),
    ("SW", r"Policies\Microsoft\Windows Defender\Real-Time Protection", "DisableRealtimeMonitoring", "REG_DWORD", "1"),
    ("SW", r"Policies\Microsoft\Windows Defender\Spynet", "SpynetReporting", "REG_DWORD", "0"),
    ("SW", r"Policies\Microsoft\Windows Defender\Spynet", "SubmitSamplesConsent", "REG_DWORD", "2"),
    ("SW", r"Policies\Microsoft\Windows\DataCollection", "AllowTelemetry", "REG_DWORD", "0"),
    ("SW", r"Policies\Microsoft\Windows\AppPrivacy", "LetAppsRunInBackground", "REG_DWORD", "2"),
    ("SW", r"Policies\Microsoft\Windows\CloudContent", "DisableWindowsConsumerFeatures", "REG_DWORD", "1"),
    ("SW", r"Policies\Microsoft\Windows\Windows Search", "AllowCortana", "REG_DWORD", "0"),
    ("SW", r"Policies\Microsoft\Dsh", "AllowNewsAndInterests", "REG_DWORD", "0"),
    ("SW", r"Policies\Microsoft\WindowsStore", "AutoDownload", "REG_DWORD", "2"),
    ("SW", r"Policies\Microsoft\EdgeUpdate", "UpdateDefault", "REG_DWORD", "0"),
    ("SW", r"Policies\Microsoft\EdgeUpdate", "AutoUpdateCheckPeriodMinutes", "REG_DWORD", "0"),
    ("SW", r"Policies\Microsoft\Edge", "StartupBoostEnabled", "REG_DWORD", "0"),
    ("SW", r"Policies\Microsoft\Edge", "BackgroundModeEnabled", "REG_DWORD", "0"),
    # visual effects: "adjust for best performance", no transparency, no animations
    ("DU", r"Software\Microsoft\Windows\CurrentVersion\Explorer\VisualEffects", "VisualFXSetting", "REG_DWORD", "2"),
    ("DU", r"Control Panel\Desktop", "UserPreferencesMask", "REG_BINARY", "9012038010000000"),
    ("DU", r"Control Panel\Desktop", "DragFullWindows", "REG_SZ", "0"),
    ("DU", r"Control Panel\Desktop", "MenuShowDelay", "REG_SZ", "0"),
    ("DU", r"Control Panel\Desktop\WindowMetrics", "MinAnimate", "REG_SZ", "0"),
    ("DU", r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize", "EnableTransparency", "REG_DWORD", "0"),
    ("DU", r"Software\Microsoft\Windows\DWM", "EnableAeroPeek", "REG_DWORD", "0"),
    ("DU", r"Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced", "TaskbarAnimations", "REG_DWORD", "0"),
    ("DU", r"Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced", "ListviewAlphaSelect", "REG_DWORD", "0"),
    ("DU", r"Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced", "ListviewShadow", "REG_DWORD", "0"),
    ("DU", r"Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced", "TaskbarDa", "REG_DWORD", "0"),
    ("DU", r"Software\Microsoft\Windows\CurrentVersion\ContentDeliveryManager", "SilentInstalledAppsEnabled", "REG_DWORD", "0"),
    ("DU", r"Software\Microsoft\Windows\CurrentVersion\ContentDeliveryManager", "SystemPaneSuggestionsEnabled", "REG_DWORD", "0"),
    ("DU", r"Software\Microsoft\Windows\CurrentVersion\ContentDeliveryManager", "SubscribedContent-338388Enabled", "REG_DWORD", "0"),
]
SLIM_HIVES = {"SYS": r"W:\Windows\System32\config\SYSTEM", "SW": r"W:\Windows\System32\config\SOFTWARE",
              "DU": r"W:\Users\Default\NTUSER.DAT"}
# run inside Windows at the first logon (things with no plain registry form)
SLIM_FIRSTLOGON = [
    "powercfg /change monitor-timeout-ac 0",
    "powercfg /change standby-timeout-ac 0",
    "powercfg /hibernate off",
    r'schtasks /change /disable /tn "\Microsoft\Windows\Defrag\ScheduledDefrag"',
    r'schtasks /change /disable /tn "\Microsoft\Windows\Application Experience\Microsoft Compatibility Appraiser"',
    r'schtasks /change /disable /tn "\Microsoft\Windows\Customer Experience Improvement Program\Consolidator"',
    r'schtasks /change /disable /tn "\Microsoft\Windows\Windows Defender\Windows Defender Scheduled Scan"',
    r'schtasks /change /disable /tn "\Microsoft\Windows\Maintenance\WinSAT"',
]


def slim_cmd():
    """Batch lines that load the applied image's hives and write SLIM_REG."""
    out = ["echo Applying the slim settings for slow PCs..."]
    for h, path in SLIM_HIVES.items():
        out.append(f'reg load HKLM\\W32{h} "{path}" >nul')
        for hh, key, name, typ, data in SLIM_REG:
            if hh == h:
                add = f'reg add "HKLM\\W32{h}\\{key}" /v "{name}" /t {typ} /d {data} /f >nul || echo   failed: {key}\\{name}'
                if "\\Services\\" in key:      # only services this build has (reg add would create a bogus key)
                    add = f'reg query "HKLM\\W32{h}\\{key}" >nul 2>&1 && ({add})'
                out.append(add)
        out.append(f"reg unload HKLM\\W32{h} >nul")
    return "\n".join(out) + "\n"


def first_logon_xml():
    return "".join(
        f'<SynchronousCommand wcm:action="add"><Order>{i + 1}</Order><CommandLine>cmd /c {c}</CommandLine></SynchronousCommand>'
        for i, c in enumerate(SLIM_FIRSTLOGON)).replace("&", "&amp;")


OOBE_XML = r"""<?xml version="1.0" encoding="utf-8"?>
<unattend xmlns="urn:schemas-microsoft-com:unattend">
  <settings pass="oobeSystem">
    <component name="Microsoft-Windows-International-Core" processorArchitecture="amd64" publicKeyToken="31bf3856ad364e35" language="neutral" versionScope="nonSxS" xmlns:wcm="http://schemas.microsoft.com/WMIConfig/2002/State">
      <InputLocale>en-US</InputLocale><SystemLocale>en-US</SystemLocale><UILanguage>en-US</UILanguage><UserLocale>en-US</UserLocale>
    </component>
    <component name="Microsoft-Windows-Shell-Setup" processorArchitecture="amd64" publicKeyToken="31bf3856ad364e35" language="neutral" versionScope="nonSxS" xmlns:wcm="http://schemas.microsoft.com/WMIConfig/2002/State">
      <OOBE>
        <HideEULAPage>true</HideEULAPage><HideOEMRegistrationScreen>true</HideOEMRegistrationScreen>
        <HideOnlineAccountScreens>true</HideOnlineAccountScreens><HideWirelessSetupInOOBE>true</HideWirelessSetupInOOBE>
        <ProtectYourPC>3</ProtectYourPC>
      </OOBE>
      <UserAccounts><LocalAccounts><LocalAccount wcm:action="add">
        <Name>User</Name><Group>Administrators</Group><Password><Value></Value><PlainText>true</PlainText></Password>
      </LocalAccount></LocalAccounts></UserAccounts>
      <FirstLogonCommands>%FIRSTLOGON%</FirstLogonCommands>
      <AutoLogon><Enabled>true</Enabled><Username>User</Username><Password><Value></Value><PlainText>true</PlainText></Password><LogonCount>9999999</LogonCount></AutoLogon>
    </component>
  </settings>
</unattend>
"""


def build_unattend(tmp, auto=False):
    """Tiny ISO 9660 image with autounattend.xml for the guest's second CD drive:
    Windows Setup reads it from removable media; it only adds Setup's LabConfig
    bypasses (the emulated PC has no TPM / Secure Boot, and < 4 GB RAM on 32-bit hosts)."""
    d = os.path.join(tmp, "unattend")
    os.makedirs(d, exist_ok=True)
    cmds = "\n".join(
        f'        <RunSynchronousCommand wcm:action="add"><Order>{i + 1}</Order>'
        f'<Path>reg add HKLM\\SYSTEM\\Setup\\LabConfig /v {k} /t REG_DWORD /d 1 /f</Path></RunSynchronousCommand>'
        for i, k in enumerate(BYPASS))
    if auto:
        d = os.path.join(tmp, "unattend-auto")
        os.makedirs(d, exist_ok=True)
        cmds = ('        <RunSynchronousCommand wcm:action="add"><Order>1</Order><Path>cmd /c for %i in '
                '(C D E F G H I J K L M N O P Q R S T U V Y Z) do @if exist %i:\\w32install.cmd call %i:\\w32install.cmd</Path>'
                '</RunSynchronousCommand>')
        with open(os.path.join(d, "w32install.cmd"), "w", newline="\r\n") as f:
            f.write(AUTO_CMD.replace("%SLIM%", slim_cmd() if SLIM else ""))
        with open(os.path.join(d, "oobe.xml"), "w", newline="\r\n") as f:
            f.write(OOBE_XML.replace("<FirstLogonCommands>%FIRSTLOGON%</FirstLogonCommands>",
                                     f"<FirstLogonCommands>{first_logon_xml()}</FirstLogonCommands>" if SLIM else ""))
    with open(os.path.join(d, "autounattend.xml"), "w", newline="\r\n") as f:
        f.write(UNATTEND % cmds)
    out = os.path.join(tmp, "unattend-auto.iso" if auto else "unattend.iso")
    r = subprocess.run(["xorriso", "-as", "mkisofs", "-quiet", "-V", "W32UNATTEND", "-J", "-o", out, d],
                       capture_output=True, text=True)
    if r.returncode:
        die("xorriso failed: " + r.stderr.strip())
    return open(out, "rb").read()


SLIM = True             # --no-slim: automatic install without the slim settings

GAP_END = 2047          # LBA of the disk stub's header; partitions must start at 2048 or later


def build_disk_stub(tmp, args):
    """Boot loader for the hard disk (installed by the emulator's install mode):
    GRUB boot.img for the MBR code area, and the gap after the MBR (LBA 1..2046)
    holding a GRUB core with an embedded menu, the emulator and the firmware.
    Module layout: 'W32ELEVEN-STUB01', u32 gap sectors | +512 boot.img | +1024 gap image."""
    files = [open(args[k], "rb").read() for k in ("emulator", "bios", "vgabios")]
    def core(locs):
        cfg = os.path.join(tmp, "disk.cfg")
        e, b, v = locs
        with open(cfg, "w") as c:
            c.write(f"""set timeout=3
set default=0
menuentry "Windows (W32Eleven x86-64 emulator)" {{
    multiboot (hd0){e[0]}+{e[1]} mode=firmware hdd=ata cdrom=optional
    module (hd0){b[0]}+{b[1]} bios
    module (hd0){v[0]}+{v[1]} vgabios
}}
""")
        out = os.path.join(tmp, "core.img")
        r = subprocess.run(["grub-mkstandalone", "-O", "i386-pc", "-o", out,
                            "--install-modules=multiboot normal configfile echo biosdisk part_msdos memdisk tar vbe video video_fb",
                            "--modules=multiboot normal configfile echo biosdisk part_msdos memdisk tar vbe video video_fb",
                            "--locales=", "--fonts=", "--themes=", f"boot/grub/grub.cfg={cfg}"], capture_output=True, text=True)
        if r.returncode:
            die("grub-mkstandalone (disk) failed: " + r.stderr.strip())
        return open(out, "rb").read()
    secs = lambda n: (n + 511) // 512
    first = 1 + secs(len(core([(0, 1)] * 3))) + 8          # slack: the real menu is a few bytes longer
    locs, lba = [], first
    for f in files:
        locs.append((lba, secs(len(f))))
        lba += secs(len(f))
    if lba > GAP_END:
        die(f"disk stub does not fit in the first MiB ({lba} sectors)")
    c = bytearray(core(locs))
    if 1 + secs(len(c)) > first:
        die("disk GRUB core grew unexpectedly")
    struct.pack_into("<QHH", c, 0x1f4, 2, secs(len(c)) - 1, 0x820)   # diskboot: rest of the core at LBA 2
    gap = bytearray((GAP_END - 1) * 512)
    gap[0:len(c)] = c
    for (l, _), f in zip(locs, files):
        gap[(l - 1) * 512:(l - 1) * 512 + len(f)] = f
    boot = open("/usr/lib/grub/i386-pc/boot.img", "rb").read()
    hdr = bytearray(512)
    hdr[0:16] = b"W32ELEVEN-STUB01"
    struct.pack_into("<I", hdr, 16, GAP_END - 1)
    return bytes(hdr) + boot + bytes(gap)


def boot_info_table(img, lba):
    """El Torito boot info table at offset 8 (GRUB's cdboot.img reads its own
    location and length from it): PVD LBA, image LBA, image length, checksum
    of the 32-bit words from offset 64 on."""
    img = bytearray(img)
    pad = (-len(img)) % 4
    words = struct.unpack_from(f"<{(len(img) + pad - 64) // 4}I", bytes(img) + b"\0" * pad, 64)
    struct.pack_into("<IIII", img, 8, 16, lba, len(img), sum(words) & 0xffffffff)
    return bytes(img)


def patch(path, args):
    with open(path, "r+b") as f:
        cat_lba = boot_catalog(f)
        cat = bytearray(read_at(f, cat_lba * SECTOR, SECTOR))
        check_validation(cat)
        entry = bytes(cat[32:64])
        size = f.seek(0, 2)
        if size % SECTOR:
            die("image size is not a multiple of 2048")
        hdr_lba = size // SECTOR
        if find_header(f, struct.unpack_from("<I", entry, 8)[0]):
            die("already patched (use --restore first)")
        if entry[0] != 0x88:
            die("the BIOS boot entry is not bootable")
        # layout after the image: header sector, emulator, BIOS, VGA BIOS, GRUB
        blobs, lba = {}, hdr_lba + 1
        with tempfile.TemporaryDirectory() as tmp0:
            extra = {"unattend": build_unattend(tmp0), "unattend_auto": build_unattend(tmp0, True),
                     "diskstub": build_disk_stub(tmp0, args)}
        for k in ("emulator", "bios", "vgabios", "unattend", "unattend_auto", "diskstub"):
            data = extra[k] if k in extra else open(args[k], "rb").read()
            blobs[k] = (lba, data)
            lba += (len(data) + SECTOR - 1) // SECTOR
        with tempfile.TemporaryDirectory() as tmp:
            img = build_grub(tmp, {k: (v[0], len(v[1])) for k, v in blobs.items()}, args["ram"], args["args"])
        img_lba = lba
        img = boot_info_table(img, img_lba)
        img += b"\0" * ((-len(img)) % SECTOR)
        hdr = bytearray(SECTOR)
        hdr[0:16] = MAGIC
        struct.pack_into("<IIQ", hdr, 16, cat_lba, img_lba, size)   # catalog LBA, GRUB image LBA, original size
        hdr[32:64] = entry                                          # original BIOS entry
        f.seek(size)
        f.write(bytes(hdr))
        for k in ("emulator", "bios", "vgabios", "unattend", "unattend_auto", "diskstub"):
            data = blobs[k][1]
            f.write(data + b"\0" * ((-len(data)) % SECTOR))
        f.write(img)
        # USB sticks: GRUB core for BIOS disks after the El Torito image, and a hybrid
        # MBR in the ISO's unused system area (sector 0) that loads it
        global BOOTDEV
        BOOTDEV = "$bootdev"
        with tempfile.TemporaryDirectory() as tmp:
            usbcore = bytearray(build_grub(tmp, {k: (v[0], len(v[1])) for k, v in blobs.items()}, args["ram"], args["args"], usb=True))
        BOOTDEV = "cd"
        usb_lba = img_lba + len(img) // SECTOR
        n512 = (len(usbcore) + 511) // 512
        struct.pack_into("<QHH", usbcore, 0x1f4, usb_lba * 4 + 1, n512 - 1, 0x820)   # diskboot: the rest of the core
        usbcore += b"\0" * ((-len(usbcore)) % SECTOR)
        f.write(bytes(usbcore))
        # GRUB's cdboot loads its core in multi-sector reads that can run past the
        # image: keep readable sectors after it (xorriso pads its images the same way)
        f.write(b"\0" * (1 << 20))
        total = f.tell()
        f.seek(0)
        orig0 = f.read(512)
        mbr = bytearray(open("/usr/lib/grub/i386-pc/boot.img", "rb").read())
        struct.pack_into("<Q", mbr, 0x5c, usb_lba * 4)                      # boot.img: core sector
        mbr[440:510] = b"\0" * 70
        mbr[446:462] = struct.pack("<BBBBBBBBII", 0x80, 0, 2, 0, 0x17, 0xfe, 0xff, 0xff, 0, total // 512)
        mbr[510:512] = b"\x55\xaa"
        f.seek(0)
        f.write(bytes(mbr))
        f.seek(size)
        hdr[512:1024] = orig0                                               # for --restore
        f.write(bytes(hdr))
        # new default entry: bootable, no emulation, load segment 0 (07C0), 4 sectors
        new = bytearray(32)
        new[0] = 0x88
        struct.pack_into("<H", new, 6, 4)
        struct.pack_into("<I", new, 8, img_lba)
        cat[32:64] = new
        f.seek(cat_lba * SECTOR)
        f.write(bytes(cat))
        print(f"patched {path}: GRUB + emulator at LBA {img_lba} ({len(img) // 1024} KiB), "
              f"original BIOS entry was LBA {struct.unpack_from('<I', entry, 8)[0]}")


def find_header(f, img_lba):
    """The header sector precedes the payload; scan back from the GRUB image."""
    for back in range(1, 4096):
        if img_lba - back < 0:
            break
        h = read_at(f, (img_lba - back) * SECTOR, SECTOR)
        if h.startswith(MAGIC) and struct.unpack_from("<I", h, 20)[0] == img_lba:
            return h
    return None


def restore(path):
    with open(path, "r+b") as f:
        cat_lba = boot_catalog(f)
        cat = bytearray(read_at(f, cat_lba * SECTOR, SECTOR))
        check_validation(cat)
        img_lba = struct.unpack_from("<I", cat, 32 + 8)[0]
        hdr = find_header(f, img_lba)
        if not hdr:
            die("not patched by w32-patch-iso")
        orig_size = struct.unpack_from("<Q", hdr, 24)[0]
        cat[32:64] = hdr[32:64]
        f.seek(0)
        f.write(hdr[512:1024])                                          # original system area sector 0
        f.seek(cat_lba * SECTOR)
        f.write(bytes(cat))
        f.truncate(orig_size)
        print(f"restored {path} ({orig_size} bytes)")


def main():
    a = sys.argv[1:]
    opts = {"emulator": "build/emulator.elf", "bios": "/usr/share/qemu/bios.bin",
            "vgabios": "/usr/share/qemu/vgabios-stdvga.bin", "ram": None, "args": ""}
    pos, mode = [], "copy"
    i = 0
    while i < len(a):
        k = a[i]
        if k in ("--emulator", "--bios", "--vgabios", "--ram", "--args"):
            if i + 1 >= len(a):
                die(f"{k} needs a value")
            opts[k[2:]] = a[i + 1]
            i += 2
            continue
        if k == "--in-place":
            mode = "inplace"
        elif k == "--restore":
            mode = "restore"
        elif k == "--no-slim":
            global SLIM
            SLIM = False
        elif k in ("-h", "--help"):
            print(__doc__)
            return
        else:
            pos.append(k)
        i += 1
    if mode == "restore":
        if len(pos) != 1:
            die("usage: --restore IMAGE.iso")
        restore(pos[0])
        return
    for k in ("emulator", "bios", "vgabios"):
        if not os.path.isfile(opts[k]):
            die(f"{k} not found: {opts[k]}")
    if not shutil.which("grub-mkstandalone"):
        die("grub-mkstandalone not found (install GRUB with i386-pc support)")
    if mode == "inplace":
        if len(pos) != 1:
            die("usage: --in-place IMAGE.iso")
        target = pos[0]
    else:
        if len(pos) != 2:
            die("usage: IN.iso OUT.iso")
        src, target = pos
        if os.path.abspath(src) == os.path.abspath(target):
            die("IN and OUT are the same file (use --in-place)")
        with open(src, "rb") as f:
            boot_catalog(f)           # fail before copying gigabytes
        with tempfile.TemporaryDirectory() as tmp:
            build_grub(tmp, {k: (0, 1) for k in ("emulator", "bios", "vgabios", "unattend", "unattend_auto", "diskstub")}, None, "")
        print(f"copying {src} -> {target} ...")
        subprocess.run(["cp", "--reflink=auto", "--sparse=always", src, target], check=True)
    patch(target, opts)


if __name__ == "__main__":
    main()
