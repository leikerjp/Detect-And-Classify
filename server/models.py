'''Defines the models for our database
   Each camera has a list of images '''

from datetime import datetime
from server import db


class Camera(db.Model):
    # value is assigned automatically because it is primary_key
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(25), nullable=False)
    # this relationship will connect a measurement to the sensor
    images = db.relationship('Image', backref='camera_node', lazy=True)

    def __repr__(self):
        return f"Camera(id='{self.id}', type='{self.name}')"


class Image(db.Model):
    # value is assigned automatically because it is primary_key
    id = db.Column(db.Integer, primary_key=True)
    date_taken = db.Column(db.DateTime, nullable=False, default=datetime.utcnow) # utcnow NOT utcnow() - function not time
    path = db.Column(db.String(250), nullable=False)
    class_name = db.Column(db.String(50), nullable=False)
    # camera_id is a column that links each image to the camera that took it (based on the cameras id)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)

    def __repr__(self):
        return f"Image(id='{self.id}, 'path='{self.path}', class_name='{self.class_name}', date_taken='{self.date_taken}', camera_id='{self.camera_id}')"