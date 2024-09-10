const cam = document.getElementById('cam')

// pegea todos os dispositivos de entrada do computador
const startVideo = () => {
    navigator.mediaDevices.enumerateDevices().then(devices => {

        // verifica se todos os dispositivos de entrada estão em um Array
        if (Array.isArray(devices)) {

            // percorre o Array de dispositivos
            devices.forEach(device => {

                // verifica se o dispositivo é uma câmera
                if (device.kind === 'videoinput') {

                    // verifica se o dispositivo é a câmera do monitor
                    if (device.label.includes('')) {
                        navigator.getUserMedia({

                            // configura o dispositivo de vídeo
                            video: {
                                deviceId: device.deviceId
                            }
                        },

                            // sucesso
                            stream => cam.srcObject = stream,
                            error => console.error(error)
                        )

                    }

                }

            })

        }
    })
}

const loadLabels = async () => {
    const labels = ['Livia Belão', 'Isabela Souza', 'Rafael Cumpri', 'Thiago Ferreira', 'Ellen De Fatima Custodio Cumpri', 'Eredio Cumpri Junior']
    const labeledDescriptors = []
    for (const label of labels) {
        const descriptions = []
        for (let i = 1; i <= 1; i++) {
            const img = await faceapi.fetchImage(`assets/lib/face-api/labels/${label}/${i}.jpg`)
            const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
            descriptions.push(detections.descriptor)
        }
        labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptions))
    }
    return labeledDescriptors
}

Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/assets/lib/face-api/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/assets/lib/face-api/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/assets/lib/face-api/models'),
    faceapi.nets.faceExpressionNet.loadFromUri('/assets/lib/face-api/models'),
    faceapi.nets.ageGenderNet.loadFromUri('/assets/lib/face-api/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/assets/lib/face-api/models')
]).then(startVideo)

cam.addEventListener('play', async () => {
    const canvas = faceapi.createCanvasFromMedia(cam)
    const canvasSize = {
        width: cam.width,
        height: cam.height
    }
    const labels = await loadLabels()
    faceapi.matchDimensions(canvas, canvasSize)
    document.body.appendChild(canvas)
    setInterval(async () => {
        const detections = await faceapi
            .detectAllFaces(
                cam,
                new faceapi.TinyFaceDetectorOptions()
            )
            .withFaceLandmarks()
            .withFaceDescriptors()
        const resizedDetection = faceapi.resizeResults(detections, canvasSize)
        const faceMatcher = new faceapi.FaceMatcher(labels, 0.5)
        const results = resizedDetection.map(d => 
            faceMatcher.findBestMatch(d.descriptor)
        )
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
        faceapi.draw.drawDetections(canvas, resizedDetection)
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetection)
        results.forEach((result, i) => {
            const box = resizedDetection[i].detection.box
            const { label, distance } = result
            new faceapi.draw.DrawTextField([
                `${label} (${parseInt(distance * 100)}%)`
            ], box.bottomLeft).draw(canvas)
        })
    }, 100)
})