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
  * @brief  ������
  * @param  ��  
  * @retval ��
  */
int main(void) 	
{		
	float frame_count = 0;
	uint8_t retry = 0;

	/* Һ����ʼ�� */
	ILI9341_Init();
	ILI9341_GramScan ( 3 );
	
	LCD_SetFont(&Font8x16);
	LCD_SetColors(RED,BLACK);

  ILI9341_Clear(0,0,320,240);	/* ��������ʾȫ�� */
	
	/********��ʾ�ַ���ʾ��*******/
  //ILI9341_DispStringLine_EN(LINE(0),"BH OV7725 Test Demo");

	USART_Config();
	LED_GPIO_Config();
	Key_GPIO_Config();
	SysTick_Init();
	
	/* ov7725 gpio ��ʼ�� */
	OV7725_GPIO_Config();
	
	LED_BLUE;
	/* ov7725 �Ĵ���Ĭ�����ó�ʼ�� */
	while(OV7725_Init() != SUCCESS)
	{
		retry++;
		if(retry>5){
			ILI9341_DispStringLine_EN(LINE(1),"No OV7725 module detected!");
			while(1);
		}
	}


	/*��������ͷ����������ģʽ*/
	OV7725_Special_Effect(cam_mode.effect);
	/*����ģʽ*/
	OV7725_Light_Mode(cam_mode.light_mode);
	/*���Ͷ�*/
	OV7725_Color_Saturation(cam_mode.saturation);
	/*���ն�*/
	OV7725_Brightness(cam_mode.brightness);
	/*�Աȶ�*/
	OV7725_Contrast(cam_mode.contrast);
	/*����Ч��*/
	OV7725_Special_Effect(cam_mode.effect);
	
	/*����ͼ�������ģʽ��С*/
	OV7725_Window_Set(cam_mode.cam_sx,
														cam_mode.cam_sy,
														cam_mode.cam_width,
														cam_mode.cam_height,
														cam_mode.QVGA_VGA);

	/* ����Һ��ɨ��ģʽ */
	ILI9341_GramScan( cam_mode.lcd_scan );
	
	//ILI9341_DispStringLine_EN(LINE(2),"OV7725 initialize success!");
	printf("\r\nOV7725����ͷ��ʼ�����\r\n");
	
	Ov7725_vsync = 0;
	
	while(1)
	{
		/*���յ���ͼ�������ʾ*/
		if( Ov7725_vsync == 2 ){
			frame_count++;
			FIFO_PREPARE;  			/*FIFO׼��*/					
			ImagDisp(cam_mode.lcd_sx,
								cam_mode.lcd_sy,
								cam_mode.cam_width,
								cam_mode.cam_height);			/*�ɼ�����ʾ*/
			
			Ov7725_vsync = 0;
		}
		
		/*��ⰴ��*/
		if( Key_Scan(KEY1_GPIO_PORT,KEY1_GPIO_PIN) == KEY_ON  ){		
			LED_BLUE;
			
			/*��ͼ�������ú�Һ����ʾ����ͽ�ͼ����*/
			ILI9341_GramScan ( cam_mode.lcd_scan );			
			
			// ��ͼ����ɺ����̵�
			Screen_Shot(110,70,LCD_X_LENGTH,LCD_Y_LENGTH);
			LED_GREEN;
			
			// ���շ���˷�����ʶ����������ʾ
			char result[20];
			char temp;
			temp = getchar();
			sprintf(result, "Result: %c", temp);
			
			ILI9341_DispStringLine_EN(LINE(1), result);
			
			// ��ͼ��ͣ���ڽص�ͼƬ�Ļ��棬ֱ������k2
			while(1){
				if( Key_Scan(KEY2_GPIO_PORT,KEY2_GPIO_PIN) == KEY_ON  )
				{
					ILI9341_Clear(0,0,100,100);
					LED_BLUE;
					break;
				}
			}
		}
		
		
		/*ÿ��һ��ʱ�����һ��֡��*/
		if(Task_Delay[0] == 0)  
		{			
			frame_count = 0;
			Task_Delay[0] = 10000;
		}
		
	}
}


/*********************************************END OF FILE**********************/

