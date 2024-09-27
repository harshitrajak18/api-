from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
        sentence = serializers.CharField()