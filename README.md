# Bangkok Building Height Estimation
Bangkok Building Height Estimation from Open Remote Sensing .... Thesis for Master of Engineering (Survey Engineering) Chulalongkorn University by Thepchai Srinoi with Advisor Associate Professor Dr. Phisan Santitamnont
![image](https://github.com/lookmeebbear/BKK_BHE/assets/88705136/a867c5a9-afa2-4177-98c0-ef0b77172854)
การประมาณความสูงอาคารด้วยข้อมูลภาพดาวเทียมแบบเปิด (โหลดมาใช้งานได้โดยไม่เสียค่าใช้จ่าย) ทั้งระบบเชิงทัศน์ Sentinel-2A/B และระบบเรดาร์ช่องเปิดสังเคราะห์ Sentinel-1A/B ใช้เป็นข้อมูลลักษณะเด่นของแบบจำลอง ประกอบกับข้อมูลความสูงอาคารจากแบบจำลองระดับพื้นผิวซึ่งเป็นผลผลิตการประมวลผลภาพจากอากาศยานไร้คนขับจากโครงการระบบสารสนเทศภูมิศาสตร์ในพื้นที่เมือง Digital Twin by 3D GIS MEA การไฟฟ้านครหลวง ใช้เป็นข้อมูลส่งออกของแบบจำลอง สร้างแบบจำลองการประมาณค่าความสูงด้วยอัลกอริทึมการเรียนรู้ด้วยเครื่องแบบ Gradient Boosting Regression เทียบกับเทคนิคที่เป็นที่นิยมก่อนหน้า Random Forest Regression กับ Support Vector Machine Regression ภายในพื้นที่เมืองกรุงเทพมหานคร
![Screenshot 2024-02-19 183110](https://github.com/lookmeebbear/BKK_BHE/assets/88705136/1f822507-90f4-44df-8206-25daa94e1d21)
ผลการทดสอบแบบจำลองที่สร้างจากข้อมูลผสมด้วยเทคนิคเกรเดียนต์บูสท์ติ้ง มีค่ารากที่สองของกำลังสองเฉลี่ยที่11.726 เมตร ที่ความสูงอาคารไม่เกิน 100 เมตร กับมีค่า 7.915 เมตร ที่ความสูงอาคารไม่เกิน 50 เมตร การใช้ข้อมูลดาวเทียมความละเอียดสิบเมตร การเลือกและคำนวณดัชนีภาพถ่าย และแบบจำลองเรียนรู้ด้วยเครื่อง ทำให้ได้ผลลัผธ์ที่จำกัดช่วงความสูงและมีสหสัมพันธ์ของคำตอบน้อย
ข้อดีของการใช้ภาพดาวเทียมประมาณค่ารังวัดคือการขยายผลการประมาณออกไปเป็นวงกว้าง ในภาพหนึ่ง scene สามารถใช้เป็นข้อมูลเบื้องต้นในทำแผนที่สามมิติ แอนิเมชันในเมือง หรือตอบปัญหาต่างๆมากมายในปัจจุบัน
![image](https://github.com/lookmeebbear/BKK_BHE/assets/88705136/893d3e42-710e-4724-bb36-18a136813f4a)


Thepchai Srinoi, Department of Survey Engineering Chulalongkorn University 2023-2024
