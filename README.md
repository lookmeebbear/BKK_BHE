# Bangkok Building Height Estimation
การประมาณความสูงอาคารจากข้อมูลดาวเทียมแบบเปิด พื้นที่ศึกษาเมืองกรุงเทพมหานคร (BUILDING HEIGHT ESTIMATION  FROM OPEN SATELLITE IMAGERY CASE STUDY IN BANGKOK URBAN AREA) วิทยานิพนธ์สำหรับปริญญาวิศวกรรมศาสตรมหาบัณฑิต สาขาวิศวกรรมสำรวจ นายเทพชัย ศรีน้อย อาจารย์ที่ปรึกษาวิทยานิพนธ์ รองศาสตราจารย์ ดร.ไพศาล สันติธรรมนนท์ 

ส่วนหนึ่งของเนื้อหาได้ตีพิมพ์เป็นบทความวิจัย Srinoi, T., Bannakulpiphat, T., Santitamnont, P., (2024) BUILDING HEIGHT ESTIMATION PRODUCTION FROM OPEN SATELLITE IMAGERY BY GRADIENT BOOSTING REGRESSION TECHNIQUE. Engineering Journal of Research and Development, Volume 35 Issue 2 April-June 2024. page 85-97 https://ph02.tci-thaijo.org/index.php/eit-researchjournal/article/view/253794

![image](https://github.com/lookmeebbear/BKK_BHE/assets/88705136/a867c5a9-afa2-4177-98c0-ef0b77172854)

การประมาณความสูงอาคารด้วยข้อมูลภาพดาวเทียมแบบเปิด (โหลดมาใช้งานได้โดยไม่เสียค่าใช้จ่าย) ทั้งระบบเชิงทัศน์ Sentinel-2A/B และระบบเรดาร์ช่องเปิดสังเคราะห์ Sentinel-1A/B ใช้เป็นข้อมูลลักษณะเด่นของแบบจำลอง ประกอบกับข้อมูลความสูงอาคารจากแบบจำลองระดับพื้นผิวซึ่งเป็นผลผลิตการประมวลผลภาพจากอากาศยานไร้คนขับจากโครงการระบบสารสนเทศภูมิศาสตร์ในพื้นที่เมือง Digital Twin by 3D GIS MEA การไฟฟ้านครหลวง ใช้เป็นข้อมูลส่งออกของแบบจำลอง สร้างแบบจำลองการประมาณค่าความสูงด้วยอัลกอริทึมการเรียนรู้ด้วยเครื่องแบบ Gradient Boosting Regression เทียบกับเทคนิคที่เป็นที่นิยมก่อนหน้า Random Forest Regression กับ Support Vector Machine Regression ภายในพื้นที่เมืองกรุงเทพมหานคร

![Screenshot 2024-02-19 183110](https://github.com/lookmeebbear/BKK_BHE/assets/88705136/1f822507-90f4-44df-8206-25daa94e1d21)

ผลการทดสอบแบบจำลองที่สร้างจากข้อมูลผสมด้วยเทคนิคเกรเดียนต์บูสท์ติ้ง มีค่ารากที่สองของกำลังสองเฉลี่ยที่11.726 เมตร ที่ความสูงอาคารไม่เกิน 100 เมตร กับมีค่า 7.915 เมตร ที่ความสูงอาคารไม่เกิน 50 เมตร การใช้ข้อมูลดาวเทียมความละเอียดสิบเมตร การเลือกและคำนวณดัชนีภาพถ่าย และแบบจำลองเรียนรู้ด้วยเครื่อง ทำให้ได้ผลลัผธ์ที่จำกัดช่วงความสูงและมีสหสัมพันธ์ของคำตอบน้อย
ข้อดีของการใช้ภาพดาวเทียมประมาณค่ารังวัดคือการขยายผลการประมาณออกไปเป็นวงกว้าง ในภาพหนึ่ง scene สามารถใช้เป็นข้อมูลเบื้องต้นในทำแผนที่สามมิติ แอนิเมชันในเมือง หรือตอบปัญหาต่างๆมากมายในปัจจุบัน

![image](https://github.com/lookmeebbear/BKK_BHE/assets/88705136/893d3e42-710e-4724-bb36-18a136813f4a)

ข้อมูลรูปอาคารจาก Microsoft Building Footprint เพิ่มความสูงจากการประมาณค่าใน Attribute Table สามารถเปิด 3D Map ใน QGIS ได้
https://drive.google.com/file/d/1HRTZ9KrcMgaasNsJiVqbTN0swLkGyyt4/view?usp=drive_link

แบบจำลองความสูงอาคารในรูปแบบ Raster Grid เหมาะกับอาคาร ไม่แนะนำวัตถุอื่นๆ ผมคิดว่าเป็นโจทย์ที่น่าสนใจ ในการศึกษาการประมาณความสูงสิ่งปกคลุมดิน Digital Surface Model Estimation ??
แนะนำให้หา Vector Layer ถนน น้ำ มาปิดส่วนที่ไม่ต้องการออก (Mask Layer)
https://drive.google.com/file/d/1GWE-KHn4gofX_xipyVTc1YHu7kxLPeoI/view?usp=drive_link



Thepchai Srinoi, Department of Survey Engineering Chulalongkorn University 2023-2024
