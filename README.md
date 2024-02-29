# VIP Filter-based Image Segmentation Tutorial

## File Structure

- `samples` Sample Images used to run the algorithms
- `output` Output images of each sample with each algorithm

## Usage Instructions
**Clone repository to local directory & change directory to project folder**
1. Create python vitrual environment
   ```python
      python3 -m venv venv
   ```
2. Activate virtual machine
   ``` python
      source venv/bin/activate
   ```
3. Install python modules
   ```python
      pip install -r requirements.txt
   ```
4. Replace the name of the desired input image in the code
   ```python
   img = cv2.imread('*image path*')
   ```
6. Run the code of each segmentation technique and observe the output 
