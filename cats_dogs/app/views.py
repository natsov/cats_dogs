import torch
from django.shortcuts import render
from torchvision import models, transforms
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.views import View
from PIL import Image
from torchvision.models import ResNet18_Weights
import os


class ImageClassifier:
    _instance = None

    def __new__(cls, model_path, device='cpu'):
        if cls._instance is None:
            cls._instance = super(ImageClassifier, cls).__new__(cls)
            cls._instance.initialize(model_path, device)
        return cls._instance

    def initialize(self, model_path, device):
        self.device = torch.device(device)
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_classes = 2
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        except RuntimeError as e:
            print(f"Model loading error: {e}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def classify_image(self, image_file):
        try:
            image = Image.open(image_file).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(image)
                _, predicted = torch.max(output, 1)
                label = 'Cat' if predicted.item() == 0 else 'Dog'
            return label
        except Exception as e:
            raise ValueError("Image processing error: " + str(e))


class ClassifyView(View):
    @method_decorator(csrf_exempt)
    def post(self, request):
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image provided'}, status=400)

        classifier = ImageClassifier(os.getenv('MODEL_PATH', 'models/model.pth'))

        try:
            label = classifier.classify_image(image_file)
            return JsonResponse({'label': label})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)


def main(request):
    return render(request, 'site/main.html', {})