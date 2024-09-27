from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import numpy as np
from .serializer import PredictionSerializer
from rest_framework.decorators import api_view
from django.conf import settings

# Load the model and vectorizer at the module level
model_path = os.path.join(settings.BASE_DIR, 'machine', 'models', 'sentiment.joblib')
vectorizer_path = os.path.join(settings.BASE_DIR, 'machine', 'models', 'vectorizer.joblib')
loaded_model = joblib.load(model_path)
vect = joblib.load(vectorizer_path))

@api_view(['POST'])
def post(request):
    serializer = PredictionSerializer(data=request.data)

    # Validate serializer data
    if not serializer.is_valid():
        return Response({
            'status': 403,
            'errors': serializer.errors,
            'message': "Validation failed"
        }, status=status.HTTP_403_FORBIDDEN)

    try:
        # Access the validated sentence
        sentence = serializer.validated_data['sentence']

        # Transform the sentence to features using the vectorizer
        features = vect.transform([sentence])

        # Make a prediction using the loaded model
        prediction = loaded_model.predict(features)

        # Return the prediction in the response
        return Response({'prediction': prediction.tolist()}, status=status.HTTP_200_OK)

    except Exception as e:
        # Handle any exceptions that occur during prediction
        return Response({
            'status': 500,
            'message': str(e),
            'error': "An error occurred during prediction"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class HomeView(APIView):
    def get(self, request):
        return Response({"message": "Welcome to the ML API!"})
