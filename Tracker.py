import math

frame_counter=40
class EuclideanDistTracker:
    def __init__(self, max_dist):
        # Store the center positions of the objects
        self.max_dist=max_dist
        self.center_points = {}
        self.counter={}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < self.max_dist:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                self.counter[self.id_count] = frame_counter
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore

        new_center_points = {}
        new_id=[]

        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id

            new_id.append(object_id)



        for id in range(self.id_count):
            if not(id in new_id):
                if self.counter[id]!=0:
                    self.counter[id]-=1
            else:
                self.counter[id]=frame_counter
            if self.counter[id]!=0:
                center = self.center_points[id]
                new_center_points[id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids


