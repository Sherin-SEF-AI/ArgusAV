`ch341.c` is the unmodified Linux USB-serial driver for the WCH CH340/CH341
(`drivers/usb/serial/ch341.c`, Linux v6.8.12, GPL-2.0, see its header). The Jetson's
kernel is built without it (`# CONFIG_USB_SERIAL_CH341 is not set`), so
`setup/usbcan-install.sh` compiles it against the installed kernel headers. It is
needed by the Waveshare USB-CAN-A adapter (`avrig/usbcan.py`).
