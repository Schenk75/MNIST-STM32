#include "stm32f10x.h"
#include "./ov7725/bsp_ov7725.h"
#include "./lcd/bsp_ili9341_lcd.h"
#include "./led/bsp_led.h"   
#include "./usart/bsp_usart.h"
#include "./key/bsp_key.h"  
#include "./systick/bsp_SysTick.h"
#include "./bmp/bsp_bmp.h"
#include <string.h>
#include <stdio.h>


extern uint8_t Ov7725_vsync;

unsigned int Task_Delay[NumOfTask]; 

extern OV7725_MODE_PARAM cam_mode;


/**
  * @brief  主函数
  * @param  无  
  * @retval 无
  */
int main(void) 	
{		
	float frame_count = 0;
	uint8_t retry = 0;

	/* 液晶初始化 */
	ILI9341_Init();
	ILI9341_GramScan ( 3 );
	
	LCD_SetFont(&Font8x16);
	LCD_SetColors(RED,BLACK);

  ILI9341_Clear(0,0,320,240);	/* 清屏，显示全黑 */
	
	/********显示字符串示例*******/
  //ILI9341_DispStringLine_EN(LINE(0),"BH OV7725 Test Demo");

	USART_Config();
	LED_GPIO_Config();
	Key_GPIO_Config();
	SysTick_Init();
	
	/* ov7725 gpio 初始化 */
	OV7725_GPIO_Config();
	
	LED_BLUE;
	/* ov7725 寄存器默认配置初始化 */
	while(OV7725_Init() != SUCCESS)
	{
		retry++;
		if(retry>5){
			ILI9341_DispStringLine_EN(LINE(1),"No OV7725 module detected!");
			while(1);
		}
	}


	/*根据摄像头参数组配置模式*/
	OV7725_Special_Effect(cam_mode.effect);
	/*光照模式*/
	OV7725_Light_Mode(cam_mode.light_mode);
	/*饱和度*/
	OV7725_Color_Saturation(cam_mode.saturation);
	/*光照度*/
	OV7725_Brightness(cam_mode.brightness);
	/*对比度*/
	OV7725_Contrast(cam_mode.contrast);
	/*特殊效果*/
	OV7725_Special_Effect(cam_mode.effect);
	
	/*设置图像采样及模式大小*/
	OV7725_Window_Set(cam_mode.cam_sx,
														cam_mode.cam_sy,
														cam_mode.cam_width,
														cam_mode.cam_height,
														cam_mode.QVGA_VGA);

	/* 设置液晶扫描模式 */
	ILI9341_GramScan( cam_mode.lcd_scan );
	
	//ILI9341_DispStringLine_EN(LINE(2),"OV7725 initialize success!");
	printf("\r\nOV7725摄像头初始化完成\r\n");
	
	Ov7725_vsync = 0;
	
	while(1)
	{
		/*接收到新图像进行显示*/
		if( Ov7725_vsync == 2 ){
			frame_count++;
			FIFO_PREPARE;  			/*FIFO准备*/					
			ImagDisp(cam_mode.lcd_sx,
								cam_mode.lcd_sy,
								cam_mode.cam_width,
								cam_mode.cam_height);			/*采集并显示*/
			
			Ov7725_vsync = 0;
		}
		
		/*检测按键*/
		if( Key_Scan(KEY1_GPIO_PORT,KEY1_GPIO_PIN) == KEY_ON  ){		
			LED_BLUE;
			
			/*截图必需设置好液晶显示方向和截图窗口*/
			ILI9341_GramScan ( cam_mode.lcd_scan );			
			
			// 截图，完成后亮绿灯
			Screen_Shot(110,70,LCD_X_LENGTH,LCD_Y_LENGTH);
			LED_GREEN;
			
			// 接收服务端发来的识别结果，并显示
			char result[20];
			char temp;
			temp = getchar();
			sprintf(result, "Result: %c", temp);
			
			ILI9341_DispStringLine_EN(LINE(1), result);
			
			// 截图后停留在截到图片的画面，直到按下k2
			while(1){
				if( Key_Scan(KEY2_GPIO_PORT,KEY2_GPIO_PIN) == KEY_ON  )
				{
					ILI9341_Clear(0,0,100,100);
					LED_BLUE;
					break;
				}
			}
		}
		
		
		/*每隔一段时间计算一次帧率*/
		if(Task_Delay[0] == 0)  
		{			
			frame_count = 0;
			Task_Delay[0] = 10000;
		}
		
	}
}


/*********************************************END OF FILE**********************/

