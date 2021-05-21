import cv2
import json
import numpy as np
import onnxruntime
import pandas as pd

from matplotlib import pyplot as plt

def parse_image(frame):
	H = (frame.shape[0]*2)//3
	W = frame.shape[1]
	parsed = np.zeros((6, H//2, W//2), dtype=np.uint8)

	parsed[0] = frame[0:H:2, 0::2]
	parsed[1] = frame[1:H:2, 0::2]
	parsed[2] = frame[0:H:2, 1::2]
	parsed[3] = frame[1:H:2, 1::2]
	parsed[4] = frame[H:H+H//4].reshape((-1, H//2,W//2))
	parsed[5] = frame[H+H//4:H+H//2].reshape((-1, H//2,W//2))

	return parsed

def seperate_odd_even(df):
	even = df.iloc[lambda x: x.index % 2 == 0]
	odd = df.iloc[lambda x: x.index % 2 != 0]
	even = pd.concat([even], ignore_index = True)
	odd = pd.concat([odd], ignore_index = True)

	return even, odd

def main():
    model = "supercombo.onnx"

    cap = cv2.VideoCapture('/Users/martinta/Documents/commai/python/data/cropped_plats.mp4')
    parsed_images = []
    x_ = np.arange(0, 33)

    width = 512
    height = 256
    dim = (width, height)

    plan_start_idx = 0
    plan_end_idx = 4955
    
    lanes_start_idx = plan_end_idx
    lanes_end_idx = lanes_start_idx + 264
    
    lane_lines_prob_start_idx = lanes_end_idx
    lane_lines_prob_end_idx = lane_lines_prob_start_idx + 8
    
    road_start_idx = lane_lines_prob_end_idx
    road_end_idx = road_start_idx + 132
    
    lead_start_idx = road_end_idx
    lead_end_idx = lead_start_idx + 55
    
    lead_prob_start_idx = lead_end_idx
    lead_prob_end_idx = lead_prob_start_idx + 3
    
    desire_start_idx = lead_prob_end_idx
    desire_end_idx = desire_start_idx + 8
    
    meta_start_idx = desire_end_idx
    meta_end_idx = meta_start_idx + 32
    
    desire_pred_start_idx = meta_end_idx
    desire_pred_end_idx = desire_pred_start_idx + 32
    
    pose_start_idx = desire_pred_end_idx
    pose_end_idx = pose_start_idx + 12
    
    rnn_start_idx = pose_end_idx
    rnn_end_idx = rnn_start_idx + 908

    session = onnxruntime.InferenceSession(model, None)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if (ret == False):
            break

        if frame is not None:
            img = cv2.resize(frame, dim)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
            parsed = parse_image(img_yuv)

            if (len(parsed_images) >= 2):
                del parsed_images[0]

            parsed_images.append(parsed)

        if (len(parsed_images) >= 2):

            parsed_arr = np.array(parsed_images)
            parsed_arr.resize((1,12,128,256))

            data = json.dumps({'data': parsed_arr.tolist()})
            data = np.array(json.loads(data)['data']).astype('float32')

            input_imgs = session.get_inputs()[0].name
            desire = session.get_inputs()[1].name
            initial_state = session.get_inputs()[2].name
            traffic_convention = session.get_inputs()[3].name
            output_name = session.get_outputs()[0].name

            desire_data = np.array([0]).astype('float32')
            desire_data.resize((1,8))

            traffic_convention_data = np.array([0]).astype('float32')
            traffic_convention_data.resize((1,512))

            initial_state_data = np.array([0]).astype('float32')
            initial_state_data.resize((1,2))

            result = session.run([output_name], {input_imgs: data,
												desire: desire_data,
												traffic_convention: traffic_convention_data,
												initial_state: initial_state_data
												})

            res = np.array(result)

            # plan = res[:,:,plan_start_idx:plan_end_idx]
            lanes = res[:,:,lanes_start_idx:lanes_end_idx]
            # lane_lines_prob = res[:,:,lane_lines_prob_start_idx:lane_lines_prob_end_idx]
            # lane_road = res[:,:,road_start_idx:road_end_idx]
            # lead = res[:,:,lead_start_idx:lead_end_idx]
   			# lead_prob = res[:,:,lead_prob_start_idx:lead_prob_end_idx]
 			# desire_state = res[:,:,desire_start_idx:desire_end_idx]
 			# meta = res[:,:,meta_start_idx:meta_end_idx]
 			# desire_pred = res[:,:,desire_pred_start_idx:desire_pred_end_idx]
 			# pose = res[:,:,pose_start_idx:pose_end_idx]
            # recurrent_layer = res[:,:,rnn_start_idx:rnn_end_idx]

            lanes_flat = lanes.flatten()
            df_lanes = pd.DataFrame(lanes_flat)
			# ll = df_lanes[0:66]
            l = df_lanes[66:132]
            r = df_lanes[132:198]
			# rr = df_lanes[198:264]

			# ll_even, ll_odd = seperate_odd_even(ll)
            l_even, l_odd = seperate_odd_even(l)
            r_even, r_odd = seperate_odd_even(r)
			# rr_even, rr_odd = seperate_odd_even(rr)

            middle = l_even.add(r_even, fill_value=0) / 2

            plt.scatter(r_even, x_, color = "b")
            plt.scatter(l_even, x_, color = "b")
            plt.scatter(middle, x_, color = "r")
            
            plt.title("Raod lines")
            plt.xlabel("blue - road lines | red - predicted path")
            plt.ylabel("Range")
            plt.show()
            plt.pause(0.1)
            plt.clf()

        frame = cv2.resize(frame, (900, 500))
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
