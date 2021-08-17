import serial 
import time
import binascii

class ser:
	def __init__(self,port,baud,timeout):
		self.port = port
		self.baud = baud
		self.timeout = timeout
		self.main_engine = None
		global Ret
		Ret = False
		try:
			self.main_engine = serial.Serial(self.port,self.baud,timeout=self.timeout)
			if self.main_engine.is_open:
				Ret = True
		except Exception as e:
			print("Open serial filed:",e)

	def openEngine(self):
		global Ret
		if not self.main_engine.is_open:
			self.main_engine.open()
			Ret = True

	def closeEngine(self):
		global Ret
		if self.main_engine.is_open:
			self.main_engine.close()
			Ret = False

	def yawControl(self,angle):
		self.main_engine.write([0xff, 0x01, 0x00, 0x51, 0x00, 0x00, 0x52])
		while True:
			count = self.main_engine.inWaiting()
			if count !=0 :
				recv = str(binascii.b2a_hex(self.main_engine.read(count)))[2:-1]
				break
			time.sleep(0.000001)
		yawAngle = int(recv[8:10],16)*256+int(recv[10:12],16)+int(angle)
		yawAngle = yawAngle-36000 if yawAngle>35999 else yawAngle if yawAngle>0 else yawAngle+36000
		print(yawAngle)
		rem = yawAngle%256
		mod = yawAngle//256
		enddata = rem+mod+75+1
		writedata = [0xff, 0x01, 0x00, 0x4b, mod, rem, enddata%256]
		self.main_engine.write(writedata)

	def pitchControl(self,angle):
		self.main_engine.write([0xff, 0x01, 0x00, 0x53, 0x00, 0x00, 0x54])
		while True:
			count = self.main_engine.inWaiting()
			if count !=0 :
				recv = str(binascii.b2a_hex(self.main_engine.read(count)))[2:-1]
				break
			time.sleep(0.000001)
		pitchAngle = int(recv[8:10],16)*256+int(recv[10:12],16)+int(angle)
		pitchAngle = pitchAngle-36000 if pitchAngle>35999 else pitchAngle if pitchAngle>0 else pitchAngle+36000
		print(pitchAngle)
		rem = pitchAngle%256
		mod = pitchAngle//256
		enddata = rem+mod+77+1
		writedata = [0xff, 0x01, 0x00, 0x4d, mod, rem, enddata%256]
		self.main_engine.write(writedata)

	def stop(self):
		self.main_engine.write([0xff, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01])

	def control(self,x1,y1,w,h,r,c):
		yaw = round((0.7906818*(w-r)/2+x1))
		pitch = round((0.7906818*(h-c)/2+y1))
		print(yaw, pitch)
		self.yawControl(yaw)
		self.pitchControl(pitch)

	def up(self):
		self.pitchControl(-7.906818*10)

	def down(self):
		self.pitchControl(7.906818*10)

	def left(self):
		self.yawControl(-7.906818*10)

	def right(self):
		self.yawControl(7.906818*10)




if __name__ == '__main__':
	# serial = serial.Serial('/dev/ttyUSB0', 19200, timeout=0.01)
	# serial.write([0xff, 0x01, 0x00, 0x51, 0x00, 0x00, 0x52])
	# while True:
	# 	count = serial.inWaiting() # 获取串口缓冲区数据
	# 	if count !=0 :
	# 		recv = str(binascii.b2a_hex(serial.read(count)))[2:-1]
	# 		break
	# 	time.sleep(0.0001)
	# print(recv[8:12])
	# addangle = 100;
	# rem = (addangle+int(recv[10:12],16))%256
	# mod = (addangle+int(recv[10:12],16))//256 + int(recv[8:10],16)
	# print(rem,mod)
	# enddata = rem+mod+75+1
	# writedata = [0xff, 0x01, 0x00, 0x4b, mod, rem, enddata%256]
	# print(writedata)
	# serial.write(writedata)
	ss = ser('com3', 19200, 0.01)
	ss.openEngine
	ss.main_engine.write([0xff, 0x01, 0x00, 0x4b, 0x00, 0x00, 0x4c])
	ss.main_engine.write([0xff, 0x01, 0x00, 0x4d, 0x00, 0x00, 0x4e])
	# ss.yawControl(100)
	# ss.pitchControl(-100)
	# ss.left();