geometry = 
0: [104.20100493728724,14.883264926892826]
1: [105.46443267166224,14.883264926892826]
2: [105.46443267166224,15.646366467759993]
3: [104.20100493728724,15.646366467759993]
4: [104.20100493728724,14.883264926892826]

var imgVV = ee.ImageCollection('COPERNICUS/S1_GRD')
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .select('VV')
//เลือกดาวเทียมเเละ polarization
var before = imgVV.filterDate('2022-05-01', '2022-05-10').mosaic().clip(geometry);
var after = imgVV.filterDate('2022-10-20', '2022-10-30').mosaic().clip(geometry);
var SMOOTHING_RADIUS = 100; 
var DIFF_UPPER_THRESHOLD = -3;
var diff_smoothed = after.focal_median(SMOOTHING_RADIUS, 'circle', 'meters')
.subtract(before.focal_median(SMOOTHING_RADIUS, 'circle', 'meters'));
var diff_thresholded = diff_smoothed.lt(DIFF_UPPER_THRESHOLD);

//เลือกวันที่เเละmosaic รูปภาพพร้อมกับตัดรูปภาพเพื่อให้อยู่ในกรอบของ geometry
Map.centerObject(geometry,13); 
//คำสั่งเมื่อกด run แล้วให้อยู่ในกรอบที่เราสนใจ
Map.addLayer(before, {min:-20,max:0,palette:['blue','green','white']}, 'Ubon Before flood');
//คำสั่งสร้าง layer ก่อนน้ำท่วมเเละใส่สีจากคำสั่ง pallette
Map.addLayer(after, {min:-20,max:0,palette:['blue','green','white']}, 'Ubon during Flood');
//คำสั่งสร้าง layer หลังน้ำท่วมเเละใส่สีจากคำสั่ง pallette
Map.addLayer(diff_thresholded.updateMask(diff_thresholded),{min:0,max:1, palette:"blue"},'flooded areas - blue',1);
var col = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(ee.Geometry.Point(105, 13.5))
  .filterDate('2020-01-01', '2020-01-31');
print('Number of images', col.size());
// Get the number of images in the collection.
