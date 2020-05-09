/**
  ******************************************************************************
  * @file    bsp_key.c
  * @author  fire
  * @version V1.0
  * @date    2013-xx-xx
  * @brief   bmp文件驱动，显示bmp图片和给液晶截图
  ******************************************************************************
  */ 
 
#include "./lcd/bsp_ili9341_lcd.h"
#include "./bmp/bsp_bmp.h"
#include "./usart/bsp_usart.h"

#define RGB24TORGB16(R,G,B) ((unsigned short int)((((R)>>3)<<11) | (((G)>>2)<<5)	| ((B)>>3)))


/**
 * @brief  设置ILI9341的截取BMP图片
 * @param  x ：截取区域的起点X坐标 
 * @param  y ：截取区域的起点Y坐标 
 * @param  Width ：区域宽度
 * @param  Height ：区域高度 
 */
void Screen_Shot( uint16_t x, uint16_t y, uint16_t Width, uint16_t Height){
	int i;
	int j;    
	unsigned char r,g,b;	
	unsigned int read_data;

	printf("++++++|[");
	
	for(i=0; i<Height; i++){
		printf("[");
		for(j=0; j<Width; j++){					
			read_data = ILI9341_GetPointPixel ( x + j, y + i );	
			
			r =  GETR_FROM_RGB16(read_data);
			g =  GETG_FROM_RGB16(read_data);
			b =  GETB_FROM_RGB16(read_data);
			if (j != Width - 1){
				printf("[%d, %d, %d], ", r, g, b);
			}
			else{
				printf("[%d, %d, %d]", r, g, b);
			}
		}
		if (i!= Height-1){
			printf("],");
		}
		else{
			printf("]");
		}
	}
	printf("]|++++++");
}
