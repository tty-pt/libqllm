/* metal.c */

#include <objc/objc.h>
#include <objc/runtime.h>
#include <objc/message.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

/*
 * Polyfill equivalente ao teu qllm_backend_mem_check(),
 * mas usando Metal em macOS.
 *
 * GPU = índice do dispositivo (0 = primeiro)
 * free_b  = VRAM livre (aproximada)
 * total_b = VRAM total do dispositivo
 *
 * Notas:
 * - Metal *não* expõe uso actual da VRAM.
 * - Só há um equivalente: registry -> IOService -> "VRAM,totalsize".
 * - A parte "free" é sempre 0 por falta de API pública.
 */

static uint64_t
metal_get_total_vram(int gpu_index)
{
	Class mtl = (Class) objc_getClass("MTLCopyAllDevices");
	if (!mtl)
		return 0;

	/* Em versões modernas: use MTLCreateSystemDefaultDevice / MTLCopyAllDevices */
	Class MTLDeviceClass = objc_getClass("MTLDevice");
	if (!MTLDeviceClass)
		return 0;

	/* MTLCopyAllDevices() */
	NSArray *(*copy_all)(void) =
		(NSArray *(*)(void)) objc_msgSend;
	id arr = copy_all((id) objc_getClass("MTLCopyAllDevices"));

	if (!arr)
		return 0;

	unsigned long count =
		(unsigned long) ((NSUInteger (*)(id, SEL)) objc_msgSend)(arr, sel_registerName("count"));
	if (gpu_index < 0 || gpu_index >= (int)count)
		return 0;

	id dev =
		((id (*)(id, SEL, NSUInteger)) objc_msgSend)(arr, sel_registerName("objectAtIndex:"), (NSUInteger) gpu_index);

	if (!dev)
		return 0;

	/* property: recommendedMaxWorkingSetSize */
	uint64_t ws =
		(uint64_t) ((uint64_t (*)(id, SEL)) objc_msgSend)(dev, sel_registerName("recommendedMaxWorkingSetSize"));

	/*
	 * Isto *não* é VRAM total real. É o working-set máximo recomendado.
	 * A alternativa real é via IOKit -> IORegistry.
	 */

	return ws;
}

#include <IOKit/IOKitLib.h>

static uint64_t
io_vram_total(void)
{
	io_iterator_t it;
	io_service_t obj;

	if (IOServiceGetMatchingServices(kIOMainPortDefault,
	    IOServiceMatching("IOPCIDevice"), &it) != KERN_SUCCESS)
		return 0;

	uint64_t vram = 0;

	while ((obj = IOIteratorNext(it))) {
		CFMutableDictionaryRef props = NULL;
		if (IORegistryEntryCreateCFProperties(obj, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS) {
			CFTypeRef val = CFDictionaryGetValue(props, CFSTR("VRAM,totalsize"));
			if (val && CFGetTypeID(val) == CFDataGetTypeID()) {
				uint64_t tmp = 0;
				CFDataGetBytes((CFDataRef)val, CFRangeMake(0, sizeof(tmp)), (UInt8 *) &tmp);
				if (tmp > vram)
					vram = tmp;
			}
			CFRelease(props);
		}
		IOObjectRelease(obj);
	}

	IOObjectRelease(it);
	return vram;
}

void
qllm_backend_mem_check(int gpu, size_t *free_b, size_t *total_b)
{
	*free_b = 0;
	*total_b = 0;

	/*
	 * Metal não tem equivalente ao VK_EXT_memory_budget.
	 * Só se obtém VRAM total via IORegistry.
	 */
	uint64_t total = io_vram_total();
	if (!total)
		total = metal_get_total_vram(gpu);

	*total_b = (size_t) total;

	/*
	 * Não existe API pública para memória livre da GPU.
	 * Metal oculta isto completamente.
	 */
	*free_b = 0;
}
